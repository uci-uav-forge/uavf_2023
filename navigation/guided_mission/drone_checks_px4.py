import rospy
import time
from py_gnc_functions import *


def drone_checks():
    # init drone api
    rate = rospy.Rate(10)
    drone = gnc_api()
    drone.wait4connect()
    drone.set_mode_px4('OFFBOARD')
    drone.initialize_local_frame()

    print('LOCAL HEADING TEST: ' + str(drone.get_current_heading()))
    #print('COMPASS HEADING TEST: ' + str(drone.get_current_compass_hdg()))
    print('LOCAL POSITION TEST: ')  
    print(drone.get_current_location())
    print('TAKEOFF TEST: ')
    drone.arm()
    drone.set_destination(
        x=0, y=0, z=10, psi=0)
    
    while not drone.check_waypoint_reached():
        print('drone has not satisfied waypoint!')
        time.sleep(1)
        pass
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)

    # run control loop
    drone_checks()
