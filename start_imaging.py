import rospy
from std_msgs.msg import Bool, String
from imaging.pipeline import Pipeline
USE_GOPRO = input("Use gopro? (y/n) ") == "y"
if input("Use real drone position and commands? (y/n) ") == "y":
    from navigation.guided_mission.py_gnc_functions import gnc_api
else:
    from navigation.mock_drone import MockDrone as gnc_api

targets_publisher = rospy.Publisher(
    name="drop_waypoints",
    data_class=String,
    queue_size=1
)
rospy.init_node("imaging_pipeline")
drone = gnc_api()
drone.initialize_local_frame()
imaging_pipeline = Pipeline(
    drone=drone, 
    img_file="gopro" if USE_GOPRO else "tests/image0.png", 
    targets_file='imaging/targets.csv',
    dry_run=False,
    drop_sub=True,
    drop_pub = targets_publisher)
print("TODO: make the camera and localizer not be mocks")
imaging_pipeline.run(num_loops=1)