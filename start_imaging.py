import rospy
from std_msgs.msg import Bool, String
import json
import time
import numpy as np
from imaging.pipeline import Pipeline
USE_GOPRO = input("Use gopro? (y/n) ") == "y"
if input("Use real drone position and commands? (y/n) ") == "y":
    from navigation.guided_mission.py_gnc_functions import gnc_api
else:
    from navigation.mock_drone import MockDrone as gnc_api

rospy.init_node("drone_GNC", anonymous=True)
targets_publisher = rospy.Publisher(
    name="drop_waypoints",
    data_class=String,
    queue_size=1
)
def start_pipeline(msg: String):
    waypoints = json.loads(msg.data)
    print(f"Got dropzone boundary: {waypoints}")
    imaging_pipeline = Pipeline(
        drone=gnc_api(), 
        img_file="gopro" if USE_GOPRO else "tests/image0_crop_smaller.png", 
        targets_file='imaging/targets.csv',
        dry_run=False,
        drop_sub=True,
        drop_pub = targets_publisher,
        drop_zone_coords=np.array(waypoints)
        )
    imaging_pipeline.run()

dropzone_boundary_sub = rospy.Subscriber(
    name="drop_boundary",
    data_class=String,
    queue_size=1,
    callback=start_pipeline
)

pub = rospy.Publisher(
    name="drop_boundary",
    data_class=String,
    queue_size=1
)

print("Publishing dropzone boundary...")
pub.publish(String(json.dumps([[0, 0], [0, 1], [1, 1], [1, 0]])))


print("Waiting for dropzone boundary...")
rospy.spin()