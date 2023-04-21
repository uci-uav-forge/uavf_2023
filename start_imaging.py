import rospy
from std_msgs.msg import Bool, String
import json
from imaging.pipeline import Pipeline
class FakeLocalizer:
    def __init__(self):
        pass
    def get_current_xyz(self):
        return (69,-1337,23)
    def get_current_pitch_roll_yaw(self):
        return (0,0,0)
    def get_current_pos_and_angles(self):
        return self.get_current_xyz(), self.get_current_pitch_roll_yaw()
USE_GOPRO = False 
img_signal = rospy.Publisher(
    name="drop_signal",
    data_class=Bool,
    queue_size=1,
) 
def receive_targets(targets: String):
    print(f"Received targets: {json.loads(targets.data)}") 
targets_subscriber = rospy.Subscriber(
    name="drop_waypoints",
    data_class=String,
    callback=receive_targets
)
targets_publisher = rospy.Publisher(
    name="drop_waypoints",
    data_class=String,
    queue_size=1
)
rospy.init_node("imaging_pipeline")
imaging_pipeline = Pipeline(
    localizer=FakeLocalizer(), 
    img_size=(5568, 4176), 
    img_file="gopro" if USE_GOPRO else "tests/image0_crop_smaller.png", 
    targets_file='imaging/targets.csv',
    dry_run=False,
    drop_sub=True,
    drop_pub = targets_publisher)
print("TODO: make the camera and localizer not be mocks")
imaging_pipeline.run(num_loops=1)