#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

import json
from imaging.pipeline import Pipeline
import rospy
from std_msgs.msg import Bool, String 
from threading import Thread
from navigation.mock_drone import MockDrone

if __name__ == "__main__":
    USE_GOPRO = False 
    
    imaging_pipeline = Pipeline(
        localizer=MockDrone(), 
        img_size=(420,69),
        img_file="gopro" if USE_GOPRO else "tests/image0_crop_smaller.png", 
        targets_file='imaging/targets.csv')
    
    def run_pipeline():
        imaging_pipeline.run(num_loops=1)
    run_pipeline()
