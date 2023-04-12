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

os.chdir("navigation")

drop_signal = rospy.Publisher(
    name="drop_signal",
    data_class=Bool,
    queue_size=1,
)

def init_mission(mission_q, use_px4=False): 
    # mission parameters in SI units
    drop_alt = 25 # m
    max_spd = 15 # m/s
    drop_spd = 5 # m/s

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
    
    alts = [wp[2] for wp in wps]
    avg_alt = np.average(alts) 
    test_map = FlightPlan(bound_coords, home, drop_alt, avg_alt)
    
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
    global_path = test_map.gen_globalpath(wps, drop_bds, tsp)

    # initialize priority queue
    for i in range(1, len(global_path)): mission_q.put((int(i), global_path[i]))

    drone = gnc_api()
    drone.wait4connect()
    if use_px4 == True:
        drone.set_mode_px4('OFFBOARD')
    else:
        drone.wait4start()
    drone.initialize_local_frame()

    return drone, global_path, drop_alt, max_spd, drop_spd, avg_alt


def mission_loop(drone, mission_q: PriorityQueue, mission_q_assigner, max_spd, drop_spd, avg_alt, dropzone_end: tuple, use_px4=False):
    # init control loop refresh rate and dropzone signal 
    rate = rospy.Rate(30)
    in_dropzone = False
    mission_q_assigner.drop_received = True
    
    if use_px4:
        drone.arm()
        drone.set_destination(
            x=0, y=0, z=avg_alt, psi=0)
    else:
        drone.takeoff(avg_alt)
    
    # initialize maximum speed
    if use_px4:
        drone.set_speed_px4(max_spd)
    else:
        drone.set_speed(max_spd)

    while not drone.check_waypoint_reached():
        pass
    
    # outer loop: check if there are more waypoints to travel to
    while mission_q.qsize() or not mission_q_assigner.drop_received:
        curr_pos = drone.get_current_location()

        # get next waypoint
        try:
            curr_wp = mission_q.queue[0][1]
        except IndexError:
            curr_wp = (curr_pos.x, curr_pos.y)

        # calc desired heading
        hdg = -90 + np.degrees(
            np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))

        # slow down and tell imaging if in dropzone
        if curr_wp == dropzone_end and not in_dropzone:
            in_dropzone = True

            bool_msg = Bool()
            bool_msg.data = in_dropzone
            drop_signal.publish(bool_msg)

            if use_px4:
                drone.set_speed_px4(drop_spd)
            else:
                drone.set_speed(drop_spd)

        # fly lower for a dropzone
        if mission_q.queue[0][0] > 1000000000 and mission_q.queue[0][0] < 2000000000:
            curr_wp = (curr_wp[0], curr_wp[1], drop_alt)

        # send command to drone
        drone.set_destination(
            x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg)

        # get next waypoint, if obs/drop detected, go there
        while not drone.check_waypoint_reached():
            next_wp = mission_q.queue[0][1]

            if next_wp != curr_wp:
                print('waypoint interrupted!\n')
                curr_wp = next_wp

                # calc desired heading
                curr_pos = drone.get_current_location()
                hdg = -90 + np.degrees(
                    np.arctan2(curr_wp[1] - curr_pos.y, curr_wp[0] - curr_pos.x))

                drone.set_destination(
                    x=curr_wp[0], y=curr_wp[1], z=curr_wp[2], psi=hdg)

            #else:
                #curr_pos = drone.get_current_location()
                #print('current position: ' + str((curr_pos.x, curr_pos.y, curr_pos.z)))
            rate.sleep()
        mission_q.get()
    
    # go to home position and land
    if use_px4:
        drone.set_speed_px4(max_spd)
    else:
        drone.set_speed(max_spd)

    drone.set_destination(
        x=0, y=0, z=avg_alt, psi=0)
    while not drone.check_waypoint_reached():
        pass
        
    drone.land()


class PriorityAssigner():
    def __init__(self, mission_q: PriorityQueue, drone: gnc_api, dropzone_end: tuple, drop_alt: int):
        self.mission_q = mission_q
        self.drone = drone
        self.dropzone_end = dropzone_end
        self.drop_alt = drop_alt
        self.drop_received = False
        
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
        prio = int(-1000000000)
        curr_pos = self.drone.get_current_location()
        
        wp_x = curr_pos.x + avoid_coord.x
        wp_y = curr_pos.y + avoid_coord.y
        wp_z = curr_pos.z + avoid_coord.z

        if self.mission_q.queue[0][0] == prio:
            self.mission_q.queue[0][1] = (wp_x, wp_y, wp_z)
        else:
            self.mission_q.put((prio, (wp_x, wp_y, wp_z)))
    

    def drop_cb(self, drop_wps):
        prio = int(1000000000)

        for i in len(drop_wps):
            wp_x = drop_wps[i][0]
            wp_y = drop_wps[i][1]
            wp_z = self.drop_alt

            add_prio = int( (wp_x - self.dropzone_end[0])**2 + (wp_y - self.dropzone_end[1])**2 )
            self.mission_q.put((prio + add_prio, (wp_x, wp_y, wp_z)))

        self.drop_received = True


def main():
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    print("initialized ROS node")

    mission_q = PriorityQueue()
    use_px4 = True

    # init mission
    drone, global_path, drop_alt, max_spd, drop_spd, avg_alt = init_mission(mission_q, use_px4)

    # init priority assigner with mission queue and dropzone wp
    drop_start = global_path[len(global_path) - 2]
    drop_end = global_path[len(global_path) - 1]
    mission_q_assigner = PriorityAssigner(mission_q, drone, drop_end, drop_alt)

    # run control loop
    print("running control loop")
    mission_loop(drone, mission_q, mission_q_assigner, max_spd, drop_spd, avg_alt, drop_end, use_px4)


class Localizer():
    def __init__(self):
        self.current_pose_g = Odometry()
        self.current_heading_g = 0.0
        self.local_offset_g = 0.0

        self.currentPos = rospy.Subscriber(
            name="mavros/global_position/local",
            data_class=Odometry,
            queue_size=1,
            callback=self.pose_cb)


    def pose_cb(self, msg):
        self.current_pose_g = msg
        self.enu_2_local()

        q0, q1, q2, q3 = (
            self.current_pose_g.pose.pose.orientation.w,
            self.current_pose_g.pose.pose.orientation.x,
            self.current_pose_g.pose.pose.orientation.y,
            self.current_pose_g.pose.pose.orientation.z,)

        psi = atan2((2 * (q0 * q3 + q1 * q2)),
                    (1 - 2 * (pow(q2, 2) + pow(q3, 2))))

        self.current_heading_g = degrees(psi) - self.local_offset_g
    

    def enu_2_local(self):
        x, y, z = (
            self.current_pose_g.pose.pose.position.x,
            self.current_pose_g.pose.pose.position.y,
            self.current_pose_g.pose.pose.position.z)

        current_pos_local = Point()
        current_pos_local.x = x * cos(radians((self.local_offset_g - 90))) - y * sin(
            radians((self.local_offset_g - 90)))
        current_pos_local.y = x * sin(radians((self.local_offset_g - 90))) + y * cos(
            radians((self.local_offset_g - 90)))
        current_pos_local.z = z

        return current_pos_local


    def get_current_heading(self):
        return self.current_heading_g


    def get_current_location(self):
        return self.enu_2_local()


if __name__ == '__main__':
    main()