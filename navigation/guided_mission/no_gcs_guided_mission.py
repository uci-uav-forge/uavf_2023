# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
# Run launch shell script or this script from the navigation directory

from queue import PriorityQueue
import numpy as np
import time
import json
import os
from enum import IntEnum
from typing import List

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
    def __init__(self, mission_q: PriorityQueue, drone: gnc_api, drop_end: tuple, drop_alt: int, gcs_url: str, flight_plan: FlightPlan):
        self.mission_q = mission_q
        self.drone = drone
        self.drop_end = drop_end
        self.drop_alt = drop_alt
        self.drop_received = False
        self.run_obs_avoid = False
        self.gcs_url=gcs_url
        self.flight_plan = flight_plan
        
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
            print((curr_pos.x, curr_pos.y))
            wp_x = curr_pos.x + avoid_coord.x
            wp_y = curr_pos.y + avoid_coord.y
            #print('received obs avoid')
            #print(time.time())
            if self.mission_q.queue[0][0] == prio:
                self.mission_q.queue[0] = (prio, (wp_x, wp_y, curr_pos.z))
            else:
                self.mission_q.put((prio, (wp_x, wp_y, curr_pos.z)))
    

    def drop_cb(self, drop_wps: String):
        print(f"Received drop waypoints: {drop_wps.data}")
        waypoints: List[List[float]] = json.loads(drop_wps.data)
        prio = WaypointPriorities.DROP_MIN_PRIORITY
        
        gps_waypoints = [
            self.flight_plan.local_to_GPS((wp[0], wp[1]))
            for wp in sorted(waypoints, key=lambda wp: wp[2])# sort by servo number
        ]

        for wp_x, wp_y, servo_num in waypoints:
            wp_z = self.drop_alt

            dist_to_drop_end = int( (wp_x - self.drop_end[0])**2 + (wp_y - self.drop_end[1])**2 )
            self.mission_q.put((prio + dist_to_drop_end, (wp_x, wp_y, wp_z, servo_num)))

        self.drop_received = True


def hdg_pos_setpoint(drone:gnc_api, wp:tuple, curr_pos:Point):
    # calc heading and send position to drone
    hdg = -90 + np.degrees(
        np.arctan2(wp[1] - curr_pos.y, wp[0] - curr_pos.x)
    )    
    drone.set_destination(     
        x=wp[0], y=wp[1], z=wp[2], psi=hdg
    )


def drop_payload(drone: gnc_api, actuator, servo_num):
    print(f"Dropping payload {servo_num}")
    for _ in range(6):
        drone.check_waypoint_reached()
        time.sleep(0.5)
    actuator.openServo(servo_num)
    for _ in range(6):
        drone.check_waypoint_reached()
        time.sleep(0.5)

def init_mission(mission_q: PriorityQueue, use_px4=False, gcs_url = "http://localhost:8000"): 
    print("waiting for mavros position message")
    if os.getenv("MOCK_DRONE") is not None:
        home = (33.646070, -117.837994) # middle earth field
        #home = (33.642608, -117.824574) # gps coordinate on arc field
    else:
        home_fix = rospy.wait_for_message('mavros/global_position/global', NavSatFix, timeout=None) 
        home = (home_fix.latitude, home_fix.longitude)
    drop_bds_publisher = rospy.Publisher(
        name='dropzone_bounds', 
        data_class=
        String, 
        queue_size=1)# IMPORTANT: DOES NOT WORK IF YOU PUT THIS CONSTRUCTOR RIGHT NEXT TO THE `drop_bds_publsiher.publish` line IDK WHY IT DROVE ME INSANE
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
    flight_plan = FlightPlan(bound_coords, home, avg_alt)
    drop_bds_local = [flight_plan.GPS_to_local((bd[0], bd[1])) for bd in drop_bds]
    drop_bds_publisher.publish(json.dumps(drop_bds_local))
    
    
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
        else: print('Not a valid option. Please try again.')

    global_path, drop_end = flight_plan.gen_globalpath(wps, drop_bds, tsp)
    print(global_path)

    # initialize priority queue and put home last
    for i in range(1, len(global_path)): mission_q.put((i, global_path[i]))
    mission_q.put((WaypointPriorities.HOME_PRIORITY, (0, 0, avg_alt)))

    drone = gnc_api()
    drone.wait4connect()
    if use_px4 == True: drone.set_mode_px4('OFFBOARD')
    else: drone.wait4start()
    drone.initialize_local_frame()
    return drone, drop_end, drop_alt, avg_alt, flight_plan


def mission_loop(drone: gnc_api, mission_q: PriorityQueue, mission_q_assigner: PriorityAssigner, max_spd, drop_spd, avg_alt, drop_end, use_px4=False, wait_for_imaging=True, gcs_url = "http://localhost:8000"):
    # init control loop refresh rate, dropzone state, payload state 
    rate = rospy.Rate(30)
    in_dropzone = False
    at_drop_end = False
    at_drop_pt  = False
    servo_num = -1
    
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
    else: drone.takeoff(avg_alt)
    
    # initialize maximum speed
    if use_px4: drone.set_speed_px4(max_spd)
    else: drone.set_speed(max_spd)
    while not drone.check_waypoint_reached():
        pass

    # change these states to turn on or off avoidance and drop reception
    mission_q_assigner.run_obs_avoid = False # True by default
    mission_q_assigner.drop_received = not wait_for_imaging # False for real mission

    # outer loop: check if there are more waypoints to travel to
    while not mission_q.empty():
        prio, curr_wp = mission_q.get()
        curr_pos = drone.get_current_location()
        
        # if only home wp is left and drop wps not received, hover
        # else get next waypoint
        if prio == WaypointPriorities.HOME_PRIORITY and not mission_q_assigner.drop_received:
            print("waiting for drop waypoints")
            curr_wp = (curr_pos.x, curr_pos.y, curr_pos.z)
            time.sleep(1)
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
        
        hdg_pos_setpoint(drone, curr_wp, curr_pos)
        print(f"Heading to waypoint {curr_wp}, priority {prio}")

        # check if waypoint has changed
        while not drone.check_waypoint_reached():
            if not mission_q.empty():
                top_prio, top_wp = mission_q.queue[0]
                
                if top_prio<0:
                    curr_pos = drone.get_current_location()
                    hdg_pos_setpoint(drone, top_wp, curr_pos)

                    while not drone.check_waypoint_reached():
                        new_top_prio, new_top_wp = mission_q.queue[0]
                        if new_top_wp != top_wp:
                            top_wp=new_top_wp
                            hdg_pos_setpoint(drone, top_wp, curr_pos)
                        rate.sleep()

                    while mission_q.queue[0][0] < 0:# clear stale obstacle avoidance waypoints
                        mission_q.get()
                    hdg_pos_setpoint(drone, curr_wp, curr_pos)
            
            # if going towards dropzone end, save status
            if curr_wp == drop_end:
                at_drop_end = True
                at_drop_pt = False
            # if going towards drop, save status and servo number
            elif WaypointPriorities.DROP_MIN_PRIORITY <= prio < WaypointPriorities.HOME_PRIORITY:
                at_drop_end = False
                at_drop_pt = True
                servo_num = curr_wp[3]
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
            drop_payload(drone, actuator, servo_num)

    mission_q_assigner.run_obs_avoid = False
    print('All waypoints reached!')
    drone.land()


def main():
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    print("initialized ROS node")
    
    mission_q = PriorityQueue()
    use_px4 = True

    # init mission
    max_spd = 10 # m/s
    drop_spd = 2 # m/s
    gcs_url = "http://localhost:8000"
    drone, drop_end, drop_alt, avg_alt, mission_plan = init_mission(mission_q, use_px4, gcs_url)

    # init priority assigner with mission queue and dropzone wp
    mission_q_assigner = PriorityAssigner(mission_q, drone, drop_end, drop_alt, gcs_url, mission_plan)

    # run online trajectory planner
    mission_loop(drone, mission_q, mission_q_assigner, max_spd, drop_spd, avg_alt, drop_end, use_px4, wait_for_imaging=False)


if __name__ == '__main__':  
    main()