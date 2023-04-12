#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

from navigation.guided_mission.py_gnc_functions import gnc_api 
import rospy
if __name__ == "__main__":
    rospy.init_node("location_test", anonymous=True)
    localizer = gnc_api()
    while 1:
        print(localizer.get_current_pos_and_angles())
