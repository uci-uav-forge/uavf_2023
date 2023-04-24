import rospy
from std_msgs.msg import String
import json

rospy.init_node("test_node", anonymous=True)
p = rospy.Publisher("drop_boundary", String, queue_size=1)
p.publish(String(json.dumps([[0, 0], [0, 1], [1, 1], [1, 0]])))