#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

import json
import rospy
from std_msgs.msg import Bool, String 
import time
if __name__ == "__main__":
    rospy.init_node("drone_GNC", anonymous=True)
    USE_GOPRO = False 
    img_signal = rospy.Publisher(
        name="drop_signal",
        data_class=Bool,
        queue_size=1,
    )
    subscriber = rospy.Subscriber(
        name="drop_signal",
        data_class=Bool,
        callback=print
    )
    bool_msg = Bool(True)
    while 1:
        if input("Press q to quit or enter to keep going\n") == 'q':
            break
        print(f"Publishing message {bool_msg.data}")
        img_signal.publish(bool_msg)
        bool_msg.data = not bool_msg.data
