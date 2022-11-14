# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
from queue import PriorityQueue
from multiprocess import Process
from multiprocess.managers import BaseManager
from tf.transformations import euler_from_quaternion
import numpy as np
import time
import json
import rospy
from sensor_msgs.msg import NavSatFix

from py_gnc_functions import *
from PrintColours import *
import sys
sys.path.insert(0, '/home/herpderk/uav_catkin_ws/src/uavf_2023/navigation/algorithms/global_path')
from flight_plan_tsp import Flight_Zone


def guided_mission(mission_q: PriorityQueue): 
    # mission parameters in SI units
    takeoff_alt = 30 # m
    drop_alt = 25 # m
    avg_spd = 20 # m/s
    drop_spd = 3

    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    home_fix = rospy.wait_for_message('mavros/global_position/global', NavSatFix, timeout=None) 
    home = (home_fix.latitude, home_fix.longitude)
    #home = (38.316376, -76.556096)

    # read mission objectives from json file
    data = json.load(open('objectives.json'))
    bound_coords = [tuple(coord) for coord in data['boundary coordinates']] 
    wps = [tuple(wp) for wp in data['waypoints']]
    drop_bds = [tuple(bd) for bd in data['drop zone bounds']]
    
    alts = [wp[2] for wp in wps]
    avg_alt = np.average(alts) 
    
    test_map = Flight_Zone(bound_coords, home, drop_alt, avg_alt)
    global_path = test_map.gen_globalpath(wps, drop_bds)
    
    # initialize drone
    drone = gnc_api()
    drone.wait4connect()
    drone.wait4start()
    # drone takeoff
    drone.initialize_local_frame()
    drone.takeoff(takeoff_alt)
    time.sleep(30)
    rate = rospy.Rate(10)
    
    # initialize priority queue
    # priority queue pops low values first
    # assignment: avoidance= d-1,000,000,000, globalpath= n in num of waypoints
    #drop= d+1,000,000,000, d for drop= distance from final waypoint(the end of the drop zone)
    for i in range(1, len(global_path)): 
        mission_q.put((int(i), global_path[i]))
    # add home position at the end, always goes last, home= 2,000,000,000
    mission_q.put((int(2000000000), (0, 0, avg_alt)))
    
    prev_wp = (0, 0, 0)
    # outer loop: check if there are more waypoints to travel to
    while mission_q.qsize():
        # get next waypoint
        curr_wp = mission_q.queue[0][1]
        print(curr_wp)
        # calc desired heading
        curr_pos = drone.enu_2_local()
        hdg = -90 + np.degrees(
            np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))

        drone.set_destination(
            x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg)

        # slow down if moving to drop point
        if mission_q.queue[0][0] > 1000000000 and mission_q.queue[0][0] < 2000000000:
            drone.set_speed(drop_spd)
        else:
            drone.set_speed(avg_spd)

        # get next waypoint, if obs/drop detected, go there
        while not drone.check_waypoint_reached():
            next_wp = mission_q.queue[0][1]

            if next_wp != curr_wp:
                print('waypoint interrupted!')
                print()
                curr_wp = next_wp
                # calc desired heading
                curr_pos = drone.enu_2_local()
                hdg = -90 + np.degrees(
                    np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))

                drone.set_destination(
                    x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg)

                # slow down if moving to drop point
                if mission_q.queue[0][0] > 1000000000 and mission_q.queue[0][0] < 2000000000:
                    drone.set_speed(drop_spd)
                else:
                    drone.set_speed(avg_spd)
            else:
                curr_pos = drone.enu_2_local()
                print('current position: ' + str((curr_pos.x, curr_pos.y)))
            rate.sleep()

        mission_q.get()
    
    # correct heading
    drone.set_destination(
        x=0, y=0, z=avg_alt, psi=0)
    time.sleep(10)
    # land at home position
    drone.land()
    

class MyManager(BaseManager):
    pass
MyManager.register('PriorityQueue', PriorityQueue)


if __name__ == '__main__':
    mission_q = PriorityQueue()
    guided_mission(mission_q)
    '''
    # initialize manager
    manager = MyManager()
    manager.start()

    mission_q = manager.PriorityQueue()
    mission_process = Process(target=guided_mission, args=[mission_q])
    '''