from queue import Queue, PriorityQueue
import time

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

from py_gnc_functions import *
from PrintColours import *
from guided_mission import init_mission, mission_loop


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


class PriorityAssigner():
    def __init__(self, mission_q: PriorityQueue, dropzone_end):
        self.mission_q = mission_q
        self.dropzone_end = dropzone_end
        self.pos_updater = Localizer()

        self.avoid_sub = rospy.Subscriber(
            name="obs_avoid_rel_coord",
            data_class=Point,
            queue_size=1,
            callback=self.avoid_cb
        )
        self.drop_sub = rospy.Subscriber(
            name="drop_coord",
            data_class=Point,
            queue_size=10,
            callback=self.drop_cb
        )
    

    def avoid_cb(self, avoid_coord):
        prio = int(-1000000000)
        curr_pos = self.pos_updater.get_current_location()
        wp_x = curr_pos.x + avoid_coord.x
        wp_y = curr_pos.y + avoid_coord.y
        wp_z = curr_pos.z + avoid_coord.z

        if self.mission_q.queue[0][0] == prio:
            self.mission_q.queue[0][1] = (wp_x, wp_y, wp_z)
        else:
            self.mission_q.put((prio, (wp_x, wp_y, wp_z)))
    

    def drop_cb(self, drop_coord):
        pass


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)

    mission_q = PriorityQueue()

    # init mission
    global_path, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt = init_mission(mission_q)

    # init priority assigner with mission queue and dropzone wp
    mission_q_assigner = PriorityAssigner(mission_q, global_path[len(global_path) - 1])

    # run control loop
    mission_loop(mission_q, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt)
