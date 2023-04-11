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
    for _i in range(num_tests):
        def log_location_callback():
            with open(f"{dir_name}/loc{index}.txt", "w") as f:
                location = drone.get_current_xyz()
                angles = drone.get_current_pitch_roll_yaw()
                f.write(f"{' '.join(map(str,location))}\n{' '.join(map(str,angles))}")
            print("location logged")
        img = cam.get_image(cb_fun=log_location_callback)
        cv.imwrite(f"{dir_name}/img{index}.png", img)
        index+=1
        print(f"image {index} taken")
    camera_tests_done = True

def goto_destination(x, y, z, psi):
    drone.set_destination(x=x, y=y, z=z, psi=psi)
    while not drone.check_waypoint_reached():
        print("drone flying to target")
        time.sleep(1)
    print("target reached")

def imaging_test_mission():
    # init drone api
    drone.wait4connect()
    drone.initialize_local_frame()

    drone.arm()
    drone.set_mode_px4('OFFBOARD')
    mission_height = 10
    goto_destination(x=0, y=0, z=mission_height, psi=0)

    camera_thread = Thread(target=capture_data, args=(5,)) 
    print("taking pics!")
    camera_thread.start()

    square_side_length = 5 
    goto_destination(x=square_side_length, y=square_side_length, z=mission_height, psi=0)
    goto_destination(x=square_side_length, y=-square_side_length, z=mission_height, psi=0)
    goto_destination(x=-square_side_length, y=-square_side_length, z=mission_height, psi=0)
    goto_destination(x=-square_side_length, y=square_side_length, z=mission_height, psi=0)
    goto_destination(x=0, y=0, z=mission_height, psi=0)

    # this is necessary so that QGroundControl doesn't see a lack of input and enter failsafe mode to land early
    while not camera_tests_done:
        print("waiting for camera tests to finish")
        drone.check_waypoint_reached()
        time.sleep(1)
    print("landing!")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    # run control loop
    imaging_test_mission()
