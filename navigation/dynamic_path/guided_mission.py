from queue import PriorityQueue
from multiprocess import Process
from multiprocess.managers import BaseManager
import numpy as np
import math
import rospy
from sensor_msgs.msg import NavSatFix

from py_gnc_functions import *
from PrintColours import *
import sys
sys.path.insert(0, '/home/herpderk/uav_catkin_ws/src/uavf_2023/navigation/algorithms/global_path')
from flight_plan_tsp import Flight_Zone


def guided_mission(mission_q: PriorityQueue): 
    # mission parameters in SI units
    takeoff_alt = 30
    drop_alt = 25
    avg_alt = 30
    avg_spd = 10
    drop_spd = 3

    # initialize ROS node and set up flight plan
    rospy.init_node("drone_GNC", anonymous=True)
    #home_fix = rospy.wait_for_message('/global_position/global', NavSatFix, timeout=None) 
    #home = (home_fix.latitude, home_fix.longitude)
    home = (38.316376, -76.556096) 
    bound_coords = [
        (38.31729702009844, -76.55617670782419), 
        (38.31594832826572, -76.55657341657302), 
        (38.31546739500083, -76.55376201277696), 
        (38.31470980862425, -76.54936361414539),
        (38.31424154692598, -76.54662761646904),
        (38.31369801280048, -76.54342380058223), 
        (38.31331079191371, -76.54109648475954), 
        (38.31529941346197, -76.54052104837133), 
        (38.31587643291039, -76.54361305817427),
        (38.31861642463319, -76.54538594175376),
        (38.31862683616554, -76.55206138505936), 
        (38.31703471119464, -76.55244787859773), 
        (38.31674255749409, -76.55294546866578),
        (38.31729702009844, -76.55617670782419),
    ]
    wps = [ #LLA, altitude is AGL
        ( 38.31652512851874,   -76.553698306299, 30), 
        (38.316930096287635,  -76.5504102489997, 30),
        ( 38.31850420404286,  -76.5520175439768, 30),
        (38.318084991945966, -76.54909120275754, 30),
        (38.317170076120384, -76.54519141386767, 30),
        ( 38.31453025427406,  -76.5446561487259, 30),
        ( 38.31534881557715, -76.54085345989367, 30),
        (38.316679010868775, -76.54884916043693, 30),
        (38.315736210982266,  -76.5453730176419, 30),
        ( 38.31603925511844, -76.54876332974675, 30)
    ]
    drop_bds = [
        (38.31461655840247, -76.54516814545798),
        (38.31442098816458, -76.54523151910101),
        (38.31440638590367, -76.54394559930905),
        (38.314208221753645, -76.54400447836372)
    ]
    test_map = Flight_Zone(bound_coords, home, drop_alt, avg_alt)
    test_map.gen_globalpath(wps, drop_bds)

    # initialize priority queue
    # priority queue pops low values first
    # assignment: avoidance= d-1000000000, target= d, path= 1000000000
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
    