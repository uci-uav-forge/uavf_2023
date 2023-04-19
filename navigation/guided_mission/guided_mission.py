# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
# Run launch shell script or this script from the navigation directory

from queue import PriorityQueue
import numpy as np
import time
import json
import os

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

from .py_gnc_functions import *
from ..global_path.flight_plan_tsp import FlightPlan
from .servo_controller import ServoController
os.chdir("navigation")


def drop_payload(actuator, servo_num):
    time.sleep(3)
    actuator.openServo(servo_num)
    time.sleep(3)


def init_mission(mission_q, use_px4=False): 
    print("waiting for mavros position message")
    home_fix = rospy.wait_for_message('mavros/global_position/global', NavSatFix, timeout=None) 
    home = (home_fix.latitude, home_fix.longitude)
    #home = (33.642608, -117.824574) # gps coordinate on arc field
    
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
    for i in range(1, len(global_path)): mission_q.put((int(i), global_path[i]))
    mission_q.put((2000000000), (0, 0, avg_alt))

    drone = gnc_api()
    drone.wait4connect()
    if use_px4 == True:
        drone.set_mode_px4('OFFBOARD')
    else:
        drone.wait4start()
    drone.initialize_local_frame()

    return drone, drop_end, drop_alt, avg_alt


def mission_loop(drone, mission_q, mission_q_assigner, actuator, max_spd, drop_spd, avg_alt, drop_end, use_px4=False):
    # init control loop refresh rate, dropzone state, payload states 
    rate = rospy.Rate(60)
    in_dropzone = False
    at_drop_pt  = False
    servo_num = -1
    mission_q_assigner.drop_received = True
    
    # init imaging signal publisher
    img_signal = rospy.Publisher(
        name="drop_signal",
        data_class=Bool,
        queue_size=1,
    )
    
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
    
    # start obstacle avoidance
    mission_q_assigner.run_obs_avoid = True

    # outer loop: check if there are more waypoints to travel to
    while mission_q.qsize():
        prio = mission_q.queue[0][0]
        curr_pos = drone.get_current_location()

        # if only home wp is left and drop wps not received, hover
        # else get next waypoint
        if prio == 2000000000 and not mission_q_assigner.drop_received:
            curr_wp = (curr_pos.x, curr_pos.y)
        else: curr_wp = mission_q.queue[0][1]

        # slow down and tell imaging if in dropzone
        if curr_wp == drop_end and not in_dropzone: 
            in_dropzone = True
            bool_msg = Bool()
            bool_msg.data = in_dropzone
            img_signal.publish(bool_msg)

            if use_px4: drone.set_speed_px4(drop_spd)
            else: drone.set_speed(drop_spd)

        # speed up if going home
        elif prio == 2000000000 and mission_q_assigner.drop_received:
            if use_px4: drone.set_speed_px4(max_spd)
            else: drone.set_speed(max_spd)
        
        # calc heading and send position to drone
        hdg = -90 + np.degrees(
            np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x)
        )    
        drone.set_destination(     
            x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg
        )

        # get next waypoint
        while not drone.check_waypoint_reached():
            prio = mission_q.queue[0][0]
            next_wp = mission_q.queue[0][1]

            # interrupt current waypoint for obstacle avoidance
            if next_wp != curr_wp:
                print('waypoint interrupted!\n')
                curr_wp = next_wp
                curr_pos = drone.get_current_location()

                # calc heading and send position to drone
                hdg = -90 + np.degrees(
                    np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x)
                )
                drone.set_destination(
                    x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg
                )
            
            # if going towards drop, save status and servo number
            if prio > 1000000000 and prio < 2000000000: 
                at_drop_pt = True
                servo_num = next_wp[3]
            else: at_drop_pt = False

            # maintain loop frequency
            rate.sleep()

        # drop payload if reached drop waypoint, pop waypoint off queue
        if at_drop_pt: drop_payload(actuator, servo_num)
        mission_q.get()

    drone.land()


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
            data_class=Float32MultiArray,
            queue_size=1,
            callback=self.drop_cb
        )
    

    def avoid_cb(self, avoid_coord):
        if self.run_obs_avoid:
            prio = int(-1000000000)
            curr_pos = self.drone.get_current_location()
            
            wp_x = curr_pos.x + avoid_coord.x
            wp_y = curr_pos.y + avoid_coord.y

            if self.mission_q.queue[0][0] == prio:
                self.mission_q.queue[0][1] = (wp_x, wp_y, curr_pos.z)
            else:
                self.mission_q.put((prio, (wp_x, wp_y, curr_pos.z)))
    

    def drop_cb(self, drop_wps):
        prio = int(1000000000)

        for i in len(drop_wps):
            wp_x = drop_wps[i][0]
            wp_y = drop_wps[i][1]
            wp_z = self.drop_alt
            servo_num = drop_wps[i][2]

            add_prio = int( (wp_x - self.drop_end[0])**2 + (wp_y - self.drop_end[1])**2 )
            self.mission_q.put((prio + add_prio, (wp_x, wp_y, wp_z, servo_num)))

        self.drop_received = True


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    print("initialized ROS node")

    mission_q = PriorityQueue()
    use_px4 = True

    # init mission
    max_spd = 15 # m/s
    drop_spd = 5 # m/s
    drone, drop_end, drop_alt, avg_alt = init_mission(mission_q, use_px4)

    # init priority assigner with mission queue and dropzone wp
    mission_q_assigner = PriorityAssigner(mission_q, drone, drop_end, drop_alt)

    # init servo actuator
    actuator = ServoController()

    # run online trajectory planner
    print("running trajectory planner")
    mission_loop(drone, mission_q, mission_q_assigner, actuator, max_spd, drop_spd, avg_alt, drop_end, use_px4)
