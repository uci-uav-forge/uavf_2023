# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
# Run launch shell script or this script from the navigation directory

from queue import PriorityQueue
import numpy as np
import time
import json
import os
from enum import IntEnum

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

if os.getenv("MOCK_DRONE") is not None:
    from ..mock_drone import MockDrone as gnc_api
else:
    from .py_gnc_functions import gnc_api
from ..global_path.flight_plan_tsp import FlightPlan
from .servo_controller import ServoController
os.chdir("navigation")

class WaypointPriorities(IntEnum):
    '''
    Obstacle avoidance has a super low number so it goes first in the queue. Drops have super high numbers so they go last, but home is even higher so it goes last.
    '''
    DROP_MIN_PRIORITY = 1000000000
    OBS_AVOID_PRIORITY = -1000000000
    HOME_PRIORITY = 2000000000

class PriorityAssigner():
    def __init__(self, mission_q: PriorityQueue, drone: gnc_api, drop_end: tuple, drop_alt: int):
        self.mission_q = mission_q
        self.drone = drone
        self.drop_end = drop_end
        self.drop_alt = drop_alt
        self.drop_received = False
        self.run_obs_avoid = False
        
        self.avoid_sub = rospy.Subscriber(
            name="obs_avoid_rel_coord",
            data_class=Point,
            queue_size=1,
            callback=self.avoid_cb
        )
        self.drop_sub = rospy.Subscriber(
            name="drop_waypoints",
            data_class=String,
            queue_size=1,
            callback=self.drop_cb
        )
    

    def avoid_cb(self, avoid_coord):
        if self.run_obs_avoid:
            prio = WaypointPriorities.OBS_AVOID_PRIORITY
            curr_pos = self.drone.get_current_location()
            
            wp_x = curr_pos.x + avoid_coord.x
            wp_y = curr_pos.y + avoid_coord.y

            if self.mission_q.queue[0][0] == prio:
                self.mission_q.queue[0][1] = (wp_x, wp_y, curr_pos.z)
            else:
                self.mission_q.put((prio, (wp_x, wp_y, curr_pos.z)))
    

    def drop_cb(self, drop_wps: str):
        waypoints: list[float] = json.loads(drop_wps)
        prio = WaypointPriorities.DROP_MIN_PRIORITY

        for wp_x, wp_y, servo_num in waypoints:
            wp_z = self.drop_alt

            dist_to_drop_end = int( (wp_x - self.drop_end[0])**2 + (wp_y - self.drop_end[1])**2 )
            self.mission_q.put((prio + dist_to_drop_end, (wp_x, wp_y, wp_z, servo_num)))

        self.drop_received = True


def drop_payload(actuator, servo_num):
    time.sleep(3)
    actuator.openServo(servo_num)
    time.sleep(3)


def init_mission(mission_q: PriorityQueue, use_px4=False): 
    print("waiting for mavros position message")
    if os.getenv("MOCK_DRONE") is not None:
        home = (33.646070, -117.837994) # middle earth field
        #home = (33.642608, -117.824574) # gps coordinate on arc field
    else:
        home_fix = rospy.wait_for_message('mavros/global_position/global', NavSatFix, timeout=None) 
        home = (home_fix.latitude, home_fix.longitude)

    
    # read mission objectives from json file
    print('\nList of mission objective files:\n')
    
    file_list = os.listdir('guided_mission/mission_objectives')
    [print('(' + str(i) + ') ' + file_list[i]) for i in range(len(file_list))]
    
    print('\nInput the number of the mission you want to load: ')
    file_num = int(input())
    data = json.load(open('guided_mission/mission_objectives/' + file_list[file_num]))
    
    bound_coords = [tuple(coord) for coord in data['boundary coordinates']] 
    wps = [tuple(wp) for wp in data['waypoints']]
    drop_bds = [tuple(bd) for bd in data['drop zone bounds']]
    
    drop_alt = drop_bds[0][2]
    alts = [wp[2] for wp in wps]
    avg_alt = np.average(alts) 
    test_map = FlightPlan(bound_coords, home, avg_alt)
    
    print('\nWould you like to reorganize the waypoints into the most efficient order? (y/n)')
    while True:
        choice = str(input())
        if choice == 'y':
            print('\nGenerating most efficient path...')
            tsp = True
            break
        elif choice == 'n':
            tsp = False
            break
        else:
            print('Not a valid option. Please try again.')

    global_path, drop_end = test_map.gen_globalpath(wps, drop_bds, tsp)
    print(global_path)

    # initialize priority queue and put home last
    for i in range(1, len(global_path)): mission_q.put((i, global_path[i]))
    mission_q.put((WaypointPriorities.HOME_PRIORITY, (0, 0, avg_alt)))

    drone = gnc_api()
    drone.wait4connect()
    if use_px4 == True:
        drone.set_mode_px4('OFFBOARD')
    else:
        drone.wait4start()
    drone.initialize_local_frame()

    return drone, drop_end, drop_alt, avg_alt


def mission_loop(drone: gnc_api, mission_q: PriorityQueue, mission_q_assigner: PriorityAssigner, max_spd, drop_spd, avg_alt, drop_end, use_px4=False, wait_for_imaging=True, run_obs_avoid=True):
    # init control loop refresh rate, dropzone state, payload state 
    rate = rospy.Rate(60)
    in_dropzone = False
    at_drop_end = False
    at_drop_pt  = False
    servo_num = -1
    
    # change these states to turn on or off avoidance and drop reception
    mission_q_assigner.run_obs_avoid = False # True by default
    mission_q_assigner.drop_received = not wait_for_imaging # False for real mission
    
    # init imaging signal publisher
    img_signal = rospy.Publisher(
        name="drop_signal",
        data_class=Bool,
        queue_size=1
    )
    bool_msg = Bool()

    # servo actuator object
    actuator = ServoController()
    
    # takeoff
    if use_px4:
        drone.arm()
        drone.set_destination(
            x=0, y=0, z=avg_alt, psi=0)
    else:
        drone.takeoff(avg_alt)
    
    # initialize maximum speed
    if use_px4: drone.set_speed_px4(max_spd)
    else: drone.set_speed(max_spd)
    while not drone.check_waypoint_reached():
        pass

    # outer loop: check if there are more waypoints to travel to
    while not mission_q.empty():
        prio, curr_wp = mission_q.queue[0]
        curr_pos = drone.get_current_location()

        # if only home wp is left and drop wps not received, hover
        # else get next waypoint
        if prio == WaypointPriorities.HOME_PRIORITY and not mission_q_assigner.drop_received:
            print("waiting for drop waypoints")
            curr_wp = (curr_pos.x, curr_pos.y, curr_pos.z)
            mission_q.put((prio, curr_wp))# enqueue home wp again

        # slow down and tell imaging if in dropzone
        if curr_wp == drop_end and not in_dropzone: 
            print("reached dropzone")
            in_dropzone = True
            bool_msg.data = True
            img_signal.publish(bool_msg)
            if use_px4: drone.set_speed_px4(drop_spd)
            else: drone.set_speed(drop_spd)

        # speed up if going home
        elif prio == WaypointPriorities.HOME_PRIORITY and mission_q_assigner.drop_received:
            if use_px4: drone.set_speed_px4(max_spd)
            else: drone.set_speed(max_spd)
        
        # calc heading and send position to drone
        hdg = -90 + np.degrees(
            np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x)
        )    
        drone.set_destination(     
            x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg
        )
        print(f'DROP END: {drop_end}')
        # check if waypoint has changed
        while not drone.check_waypoint_reached():
            top_prio, top_wp = mission_q.queue[0]
            # these should be the same as prio and curr_wp unless PriorityAssigner has enqueued an obstacle avoidance waypoint in between the top of the loop and here

            # interrupt current waypoint for obstacle avoidance
            if top_wp != curr_wp:
                print('waypoint interrupted!\n')
                curr_pos = drone.get_current_location()

                # calc heading and send position to drone
                hdg = -90 + np.degrees(
                    np.arctan2(top_wp[1] - top_wp.y, top_wp[0] - curr_pos.x)
                )
                drone.set_destination(
                    x=top_wp[0], y=top_wp[1], z=top_wp[2], psi=hdg
                )
            
            # if going towards dropzone end, save status
            if top_wp == drop_end:
                at_drop_end = True
                at_drop_pt = False
            # if going towards drop, save status and servo number
            elif WaypointPriorities.DROP_MIN_PRIORITY < prio < WaypointPriorities.HOME_PRIORITY:
                print("going toward drop waypoint") 
                at_drop_end = False
                at_drop_pt = True
                servo_num = top_wp[3]
            else: 
                at_drop_end = False
                at_drop_pt = False

            # maintain loop frequency
            rate.sleep()
        # tell imaging to stop taking photos when dropzone end reached
        if at_drop_end: 
            print("Signaling imaging to stop")
            bool_msg.data = False
            img_signal.publish(bool_msg)
            at_drop_end=False
        # drop payload if reached drop waypoint
        elif at_drop_pt: 
            drop_payload(actuator, servo_num)
        #pop waypoint off queue
        mission_q.get()

    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    print("initialized ROS node")

    mission_q = PriorityQueue()
    use_px4 = True

    # init mission
    max_spd = 7 # m/s
    drop_spd = 3 # m/s
    drone, drop_end, drop_alt, avg_alt = init_mission(mission_q, use_px4)

    # init priority assigner with mission queue and dropzone wp
    mission_q_assigner = PriorityAssigner(mission_q, drone, drop_end, drop_alt)

    # run online trajectory planner
    print("running trajectory planner")
    mission_loop(drone, mission_q, mission_q_assigner, max_spd, drop_spd, avg_alt, drop_end, use_px4, wait_for_imaging=True, run_obs_avoid=False)
