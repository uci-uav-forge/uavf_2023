import rospy
from py_gnc_functions import *


def drone_checks():
    # init drone api
    rate = rospy.Rate(10)
    drone = gnc_api()
    drone.wait4connect()
    drone.set_mode_px4('OFFBOARD')
    
    print('LOCAL HEADING TEST: ' + str(drone.get_current_heading()))
    print('COMPASS HEADING TEST: ' + str(drone.get_current_compass_hdg()))
    print('LOCAL POSITION TEST: ' + str(drone.get_current_location()))
    print('ARMING TEST: ')
    drone.arm()
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)

    # run control loop
    drone_checks()
