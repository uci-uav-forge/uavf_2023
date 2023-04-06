#!/usr/bin/python3
"""
Master script to run Imaging and Navigation pipelines.
Currently only running Imaging pipeline to test for Feb. 16 flight day.
"""
from .navigation.guided_mission.guided_mission import init_mission, mission_loop
from imaging.pipeline import Pipeline
from queue import PriorityQueue
import time
import rospy
import threading

def do_imaging(drone):
    imaging_pipeline = Pipeline(drone, (5568, 4176), img_file="gopro", targets_file='imaging/targets.csv')
    imaging_pipeline.run(num_loops=50)

if __name__ == "__main__":
    rospy.init_node("drone_GNC", anonymous=True)
    print("initialized ROS node")

    mission_q = PriorityQueue()
    use_px4 = True
    drone, global_path, drop_alt, max_spd, drop_spd, avg_alt = init_mission(mission_q, use_px4)

    img_thread = threading.Thread(target = do_imaging, args=[drone])

    img_thread.run()

    mission_loop(drone, mission_q, max_spd, drop_spd, avg_alt, use_px4)
