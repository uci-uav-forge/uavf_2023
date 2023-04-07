#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""
from std_msgs.msg import *
from .navigation.guided_mission.py_gnc_functions import loc_api
from imaging.pipeline import Pipeline
from queue import PriorityQueue
import time
import rospy
import threading

if __name__ == "__main__":
    rospy.init_node("drone_IMAGING", anonymous=True)

    drop_pub = rospy.Publisher(
            name="drop_waypoints",
            data_class=Float32MultiArray,
            queue_size=1)
    
    imaging_pipeline = Pipeline(loc_api(), (5568, 4176), drop_pub, img_file="gopro", targets_file='imaging/targets.csv')
    imaging_pipeline.run(num_loops=50)
