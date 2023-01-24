from queue import Queue, PriorityQueue
import time
import rospy
from geometry_msgs.msg import Point

from py_gnc_functions import *
from PrintColours import *
from guided_mission import init_mission, mission_loop
from run_mission import Localizer


def dummy_obs_avoid():
    obs_avoid_pub = rospy.Publisher(
        name="obs_avoid_rel_coord",
        data_class=Point,
        queue_size=1
    )

    wait_time = 60
    st = time.time()
    et = time.time()

    while(et - st < wait_time):
        et = time.time()
        print('time until obstacle avoidance: ' + str(round(wait_time - et + st)))
        print()
        time.sleep(1)

    avoid_coord = Point()
    avoid_coord.x = 13.85
    avoid_coord.y = 8.00
    avoid_coord.z = 1.00
    obs_avoid_pub.publish(avoid_coord)


if __name__ == '__main__':
    rospy.init_node("dummy_obs_avoid", anonymous=True)
    try:
        dummy_obs_avoid()
    except KeyboardInterrupt:
        pass
        