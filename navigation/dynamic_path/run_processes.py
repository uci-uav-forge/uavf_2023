from multiprocess import Process
from multiprocess.managers import BaseManager
from queue import Queue, PriorityQueue
import time
import rospy
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
            callback=self.pose_cb,
        )


    def pose_cb(self, msg):
        """Gets the raw pose of the drone and processes it for use in control.

        Args:
                msg (geometry_msgs/Pose): Raw pose of the drone.
        """
        self.current_pose_g = msg
        self.enu_2_local()

        q0, q1, q2, q3 = (
            self.current_pose_g.pose.pose.orientation.w,
            self.current_pose_g.pose.pose.orientation.x,
            self.current_pose_g.pose.pose.orientation.y,
            self.current_pose_g.pose.pose.orientation.z,
        )

        psi = atan2((2 * (q0 * q3 + q1 * q2)),
                    (1 - 2 * (pow(q2, 2) + pow(q3, 2))))

        self.current_heading_g = degrees(psi) - self.local_offset_g
    

    def enu_2_local(self):
        x, y, z = (
            self.current_pose_g.pose.pose.position.x,
            self.current_pose_g.pose.pose.position.y,
            self.current_pose_g.pose.pose.position.z,
        )

        current_pos_local = Point()

        current_pos_local.x = x * cos(radians((self.local_offset_g - 90))) - y * sin(
            radians((self.local_offset_g - 90))
        )

        current_pos_local.y = x * sin(radians((self.local_offset_g - 90))) + y * cos(
            radians((self.local_offset_g - 90))
        )

        current_pos_local.z = z

        return current_pos_local


    def get_current_heading(self):
        """Returns the current heading of the drone.

        Returns:
            Heading (Float): θ in is degrees.
        """
        return self.current_heading_g


    def get_current_location(self):
        """Returns the current position of the drone.

        Returns:
            Position (geometry_msgs.Point()): Returns position of type geometry_msgs.Point().
        """
        return self.enu_2_local()


# priority queue pops low values first
# assignment: avoidance= -1,000,000,000, globalpath= n in num of waypoints
#drop= d+1,000,000,000, d for drop= distance from final waypoint(the end of the drop zone)
def prio_assign(avoid_q: Queue, drop_q: Queue, mission_q: PriorityQueue, global_path: list):
    try:
        while True:

            while avoid_q.qsize():
                avoid_wp = avoid_q.get()
                prio = int(-1000000000)
                # only want 1 obs avoid wp in the queue at a time
                if mission_q.queue[0][0] == prio:
                    mission_q.queue[0][1] = avoid_wp
                else:
                    mission_q.put((prio, avoid_wp))

            # imaging gonna give enu, we don't know if payload 
            # can change the order or keep it static, 
            # drop alt assigned in mission loop
            if drop_q.qsize():
                drop_info = dropq.get()

    except KeyboardInterrupt:
        pass

def obs_avoid(avoid_q: Queue):
    pos_updater = Localizer()
    time.sleep(120)
    curr_pos = pos_updater.get_current_location()
    avoid_wp = (curr_pos.x + 13.85, curr_pos.y + 8.00)
    avoid_q.put(avoid_wp)
    

def drop_detect(drop_q: Queue):
    try:
        pos_updater = Localizer()

    except KeyboardInterrupt:
        pass


class MyManager(BaseManager):
    pass
MyManager.register('PriorityQueue', PriorityQueue)
MyManager.register('Queue', Queue)


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    
    # init manager
    manager = MyManager()
    manager.start()

    # init queues
    avoid_q = manager.Queue()
    drop_q = manager.Queue()
    mission_q = manager.PriorityQueue()

    # init mission
    global_path, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt = init_mission(mission_q)

    # make processes
    mission_proc = Process(target=mission_loop, args=[mission_q, takeoff_alt, drop_alt, avg_spd, drop_spd, avg_alt])
    assign_proc = Process(target=prio_assign, args=[avoid_q, drop_q, mission_q, global_path])
    avoid_proc = Process(target=obs_avoid, args=[avoid_q])
    drop_proc = Process(target=drop_detect, args=[drop_q])
    
    # run processes
    mission_proc.start()
    assign_proc.start()
    avoid_proc.start()
    drop_proc.start()

   # wait until processes finish
    mission_proc.join()
    assign_proc.join()
    avoid_proc.join()
    drop_proc.join()

    manager.shutdown()
