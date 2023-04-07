#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

from imaging.pipeline import Pipeline
from navigation.guided_mission.guided_mission import Localizer
import rospy

if __name__ == "__main__":
    rospy.init_node("imaging_pipeline", anonymous=True)
    localizer = Localizer()
    imaging_pipeline = Pipeline(localizer, (5568, 4176), img_file="gopro")
    imaging_pipeline.run(num_loops=2)
