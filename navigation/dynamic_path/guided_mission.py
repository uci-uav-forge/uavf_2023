# Implementation of guidance, navigation, and control using Ardupilot, MAVROS, and the Intelligent-Quads GNC package!
from queue import PriorityQueue
from multiprocess import Process
from multiprocess.managers import BaseManager
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
    avg_spd = 10 # m/s
    drop_spd = 3

    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    #home_fix = rospy.wait_for_message('/global_position/global', NavSatFix, timeout=None) 
    #home = (home_fix.latitude, home_fix.longitude)
    home = (38.316376, -76.556096)

    # read mission objectives from json file
    data = json.load(open('objectives.json'))
    bound_coords = [tuple(coord) for coord in data['boundary coordinates']] 
    wps = [tuple(wp) for wp in data['waypoints']]
    drop_bds = [tuple(bd) for bd in data['drop zone bounds']]
    
    alts = [wp[2] for wp in wps]
    avg_alt = sum(alts)/len(alts) 
    
    test_map = Flight_Zone(bound_coords, home, drop_alt, avg_alt)
    test_map.gen_globalpath(wps, drop_bds)

    # initialize priority queue
    # priority queue pops low values first
    # assignment: avoidance= d-1000000000, globalpath= 0, drop= d+1000000000, d=distance from final waypoint(the end of the drop zone)
    for i in range(1, len(test_map.global_path)): 
        mission_q.put((1000000000, test_map.global_path[i]))
    
    # initialize drone
    drone = gnc_api()
    drone.wait4connect()
    drone.wait4start()
    # drone takeoff
    drone.initialize_local_frame()
    drone.takeoff(takeoff_alt)
    rate = rospy.Rate(10)
    
    # outer loop: check if there are more waypoints to travel to
    while mission_q.qsize():
        curr_wp = mission_q.queue[0][1]
        #align psi with line from curr_pos to curr_wp
        drone.set_destination(
            x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=???
        )
        # slow down if moving to drop point
        if mission_q.queue[0][0] > 0 and mission_q.queue[0][0] < 1000000000:
            drone.set_speed(drop_spd)
        else:
            drone.set_speed(avg_spd)

        # get next waypoint, if obs/drop detected, go there
        while not drone.check_waypoint_reached():
            next_wp = mission_q.queue[0][1]

            if next_wp != curr_wp:
                curr_wp = next_wp
                #align psi with line from curr_pos to curr_wp
                drone.set_destination(
                    x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=???
                )
                # slow down if moving to drop point
                if mission_q.queue[0][0] > 0 and mission_q.queue[0][0] < 1000000000:
                    drone.set_speed(drop_spd)
                else:
                    drone.set_speed(avg_spd)
                rate.sleep()

        mission_q.get()
    

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