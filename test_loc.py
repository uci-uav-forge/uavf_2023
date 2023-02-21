"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""

from navigation.guided_mission.run_mission import Localizer
import rospy
if __name__ == "__main__":
    rospy.init_node("imaging_pipeline", anonymous=True)
    localizer = Localizer()
    while 1:
        print(localizer.get_current_location())
