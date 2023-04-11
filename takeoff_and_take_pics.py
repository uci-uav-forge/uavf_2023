import rospy
import time
from navigation.guided_mission.py_gnc_functions import *
from imaging.camera import GoProCamera
import os
import cv2 as cv
from threading import Thread

cam = GoProCamera()
dir_name = f"../gopro_tests/{time.strftime(r'%m-%d|%H_%M_%S')}"
os.makedirs(dir_name, exist_ok=True)
drone = gnc_api()
index = 0
camera_tests_done = False
def capture_data(num_tests):
    global index, camera_tests_done
    print(index)
    for i in range(num_tests):
        img = cam.get_image()
        cv.imwrite(f"{dir_name}/img{index}.png", img)
        with open(f"{dir_name}/loc{index}.txt", "w") as f:
            location = drone.get_current_xyz()
            angles = drone.get_current_pitch_roll_yaw()
            f.write(f"{' '.join(map(str,location))}\n{' '.join(map(str,angles))}")
        index+=1
        print(f"image {index} taken")
    camera_tests_done = True

def imaging_test_mission():
    # init drone api
    drone.wait4connect()
    drone.set_mode_px4('OFFBOARD')
    drone.initialize_local_frame()

    drone.arm()
    drone.set_destination(
        x=0, y=0, z=23, psi=0)
    while not drone.check_waypoint_reached():
        print('drone has not satisfied waypoint!')
        time.sleep(1)

    camera_thread = Thread(target=capture_data, args=(2,)) 
    print("taking pics!")
    camera_thread.start()

    # this is necessary so that QGroundControl doesn't see a lack of input and enter failsafe mode to land early
    while not camera_tests_done:
        print("hovering")
        drone.check_waypoint_reached()
        time.sleep(1)
    print("landing!")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    # run control loop
    imaging_test_mission()
