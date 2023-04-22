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
    rospy.init_node("drone_GNC", anonymous=True)
    USE_GOPRO = False 
    img_signal = rospy.Publisher(
        name="drop_signal",
        data_class=Bool,
        queue_size=1,
    ) 
    def receive_targets(targets: String):
        print(f"Received targets: {json.loads(targets.data)}") 
    targets_subscriber = rospy.Subscriber(
        name="targets",
        data_class=String,
        callback=receive_targets
    )
    targets_publisher = rospy.Publisher(
        name="targets",
        data_class=String,
        queue_size=1
    )
    imaging_pipeline = Pipeline(
        localizer=MockDrone(), 
        img_file="gopro" if USE_GOPRO else "tests/image0_crop_smaller.png", 
        targets_file='imaging/targets.csv',
        dry_run=False,
        drop_sub=True,
        drop_pub = targets_publisher)
    
    def run_pipeline():
        imaging_pipeline.run(num_loops=1)

    pipeline_thread = Thread(target=run_pipeline)
    pipeline_thread.start()
    input("Press enter to send drop signal")
    start_msg=Bool(True)
    img_signal.publish(start_msg)
    input("Press enter again to stop dropping")
    stop_msg=Bool(False)
    img_signal.publish(stop_msg)
    print("Sent drop stop message")
