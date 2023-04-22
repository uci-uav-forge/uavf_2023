import rospy
from std_msgs.msg import Bool, String
import json
from imaging.pipeline import Pipeline
from navigation.mock_drone import MockDrone

USE_GOPRO = False 
img_signal = rospy.Publisher(
    name="drop_signal",
    data_class=Bool,
    queue_size=1,
) 

targets_publisher = rospy.Publisher(
    name="drop_waypoints",
    data_class=String,
    queue_size=1
)
rospy.init_node("imaging_pipeline")
imaging_pipeline = Pipeline(
    drone=MockDrone(), 
    img_file="gopro" if USE_GOPRO else "tests/image0_crop_smaller.png", 
    targets_file='imaging/targets.csv',
    dry_run=False,
    drop_sub=True,
    drop_pub = targets_publisher)
print("TODO: make the camera and localizer not be mocks")
imaging_pipeline.run(num_loops=1)