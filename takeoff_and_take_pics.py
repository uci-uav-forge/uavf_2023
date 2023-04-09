import rospy
import time
from navigation.guided_mission.py_gnc_functions import *
from imaging.camera import GoProCamera
import os
import cv2 as cv
from threading import Thread

cam = GoProCamera()
dir_name = f"gopro_tests/{time.strftime(r'%m-%d|%H_%M_%S')}"
os.makedirs(dir_name, exist_ok=True)
drone = gnc_api()
index = 0
def camera_tests(num_tests):
    global index
    print(index)
    for i in range(num_tests):
        img = cam.get_image()
        cv.imwrite(f"{dir_name}/img{index}.png", img)
        with open(f"{dir_name}/loc{index}.txt", "w") as f:
            location = drone.get_current_xyz()
            angles = drone.get_current_pitch_roll_yaw()
            f.write(f"{' '.join(map(str,location))}\n{' '.join(map(str,angles))}")
        index+=1

def imaging_test_mission():
    # init drone api
    drone.wait4connect()
    drone.set_mode_px4('OFFBOARD')
    drone.initialize_local_frame()

    print('LOCAL HEADING TEST: ' + str(drone.get_current_heading()))
    #print('COMPASS HEADING TEST: ' + str(drone.get_current_compass_hdg()))
    print('LOCAL POSITION TEST: ')  
    print(drone.get_current_location())
    print('TAKEOFF TEST: ')
    print(drone.get_current_xyz())
    print(drone.get_current_pitch_roll_yaw())
    drone.arm()
    drone.set_destination(
        x=0, y=0, z=10, psi=0)
    thread = Thread(target=camera_tests, args=(5,)) 
    while not drone.check_waypoint_reached():
        print('drone has not satisfied waypoint!')
        time.sleep(1)
    print("taking pics!")
    thread.start()
    for i in range(20):
        print("pausing")
        drone.check_waypoint_reached()
        time.sleep(1)
    print("landing!")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    camera_tests(1)
    # run control loop
    imaging_test_mission()
