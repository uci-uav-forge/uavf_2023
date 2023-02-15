# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
from queue import PriorityQueue
# from multiprocess.managers import BaseManager
import numpy as np
import time
import json

import rospy
from sensor_msgs.msg import NavSatFix

from py_gnc_functions import *
from PrintColours import *
import sys
sys.path.append("..")
from global_path.flight_plan_tsp import FlightPlan


def init_mission(mission_q): 
    # mission parameters in SI units
    takeoff_alt = 30 # m
    drop_alt = 25 # m
    avg_spd = 15 # m/s
    drop_spd = 3 # m/s

    home_fix = rospy.wait_for_message('mavros/global_position/global', NavSatFix, timeout=None) 
    home = (home_fix.latitude, home_fix.longitude)

    # read mission objectives from json file
    data = json.load(open('objectives.json'))
    bound_coords = [tuple(coord) for coord in data['boundary coordinates']] 
    wps = [tuple(wp) for wp in data['waypoints']]
    drop_bds = [tuple(bd) for bd in data['drop zone bounds']]
    
    alts = [wp[2] for wp in wps]
    avg_alt = np.average(alts) 
    
    test_map = FlightPlan(bound_coords, home, drop_alt, avg_alt)
    global_path = test_map.gen_globalpath(wps, drop_bds)

    # initialize priority queue
    for i in range(1, len(global_path)): 
        mission_q.put((int(i), global_path[i]))
    # add home position at the end, always goes last, home= 2,000,000,000
    mission_q.put((int(2000000000), (0.0, 0.0, avg_alt)))

    return global_path, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt


def mission_loop(mission_q: PriorityQueue, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt):
    # init drone api
    rate = rospy.Rate(10)
    drone = gnc_api()
    drone.wait4connect()
    drone.wait4start()

    # drone takeoff
    drone.initialize_local_frame()
    drone.takeoff(takeoff_alt)
    drone.set_position_target(
        x=0, y=0, z=takeoff_alt, psi=0, speed_mps=avg_spd)
    while not drone.check_waypoint_reached():
        pass
    
    prev_wp = (0, 0, 0)
    # outer loop: check if there are more waypoints to travel to
    while mission_q.qsize():
        # get next waypoint
        curr_wp = mission_q.queue[0][1]
        # calc desired heading
        curr_pos = drone.enu_2_local()
        hdg = -90 + np.degrees(
            np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))
        x, y, z=curr_wp[0], curr_wp[1], curr_wp[2]
        # lower alt and slow down if moving to drop point
        if mission_q.queue[0][0] > 1000000000 and mission_q.queue[0][0] < 2000000000:
            curr_wp = (curr_wp[0], curr_wp[1], drop_alt)
            # drone.set_speed(drop_spd)
            drone.set_position_target(x, y, z, hdg, drop_spd)
        else:
            drone.set_position_target(x, y, z, hdg, avg_spd)


        # get next waypoint, if obs/drop detected, go there
        while not drone.check_waypoint_reached():
            next_wp = mission_q.queue[0][1]
            if next_wp != curr_wp:
                print('waypoint interrupted!')
                print()
                curr_wp = next_wp
                # calc desired heading
                curr_pos = drone.get_current_location()
                hdg = -90 + np.degrees(
                    np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))

                # lower alt and slow down if moving to drop point
                if mission_q.queue[0][0] > 1000000000 and mission_q.queue[0][0] < 2000000000:
                    curr_wp = (curr_wp[0], curr_wp[1], drop_alt)
                    speed = drop_spd
                else:
                    speed = avg_spd
                #     drone.set_speed(avg_spd)
                # drone.set_destination(
                #     x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg)
                drone.set_position_target(x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg, speed_mps=speed)

            else:
                curr_pos = drone.get_current_location()
                print('current position: ' + str((curr_pos.x, curr_pos.y)))
            rate.sleep()

        mission_q.get()
    
    # correct heading
    drone.set_position_target(
        x=0, y=0, z=0, psi=hdg, speed_mps=avg_spd)
    time.sleep(10)
    # land at home position
    drone.land()
    

if __name__ == '__main__':
    #rospy.init_node("drone_GNC", anonymous=True)

    mission_q = PriorityQueue()
    # init mission
    global_path, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt = init_mission(mission_q)
    # run control loop
    mission_loop(mission_q, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt)
