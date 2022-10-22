from mavros_msgs.msg import GlobalPositionTarget
from geographic_msgs.msg import GeoPoseStamped
import rospy
from py_gnc_functions import *

class GPS_Publisher():
    
    def __init__(self):
        rospy.init_node('gps_waypoint_publisher', anonymous=True) 
        self.gps_pub = rospy.Publisher(
            '/mavros/setpoint_position/global', GeoPoseStamped, queue_size=10
        )

        self.drone = gnc_api()
        self.drone.wait4connect()
        self.drone.wait4start()
        self.drone.takeoff(5)
    

    def set_waypoint(self, lat, long):
        '''
        msg = GlobalPositionTarget()
        msg.latitude = lat
        msg.longitude = long
        msg.altitude = 5
        '''
        msg = GeoPoseStamped()
        msg.pose.position.latitude = lat
        msg.pose.position.longitude = long
        msg.pose.position.altitude = 5
        self.gps_pub.publish(msg)
        #self.drone.land()


if __name__ == '__main__':
    new_pub = GPS_Publisher()
    new_pub.set_waypoint(-30.36226029, 160.16329794)
