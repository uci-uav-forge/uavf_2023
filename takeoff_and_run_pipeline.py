import rospy
import time
from navigation.guided_mission.py_gnc_functions import *
from imaging.pipeline import Pipeline
import os
import cv2 as cv
from threading import Thread

dir_name = f"gopro_tests/{time.strftime(r'%m-%d|%H_%M_%S')}"
os.makedirs(dir_name, exist_ok=True)
drone = gnc_api()

pipeline = Pipeline(
    localizer = drone,
    img_size = (5568, 4176),
    img_file = "gopro",
    targets_file = "test_mission_targets.csv",
    dry_run = False
)

target_coords = None
processing_done = False

def run_pipeline():
    global target_coords, processing_done
    pipeline.run(num_loops=1)
    target_coords = pipeline.target_aggregator.get_target_coords()[0]
    processing_done = True
    print("done running pipeline")

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
    while not drone.check_waypoint_reached():
        print('drone has not satisfied waypoint!')
        time.sleep(1)

    camera_thread = Thread(target=run_pipeline) 
    print("taking pics!")
    camera_thread.start()

    # this is necessary so that QGroundControl doesn't see a lack of input and enter failsafe mode to land early
    while not processing_done:
        print("hovering")
        drone.check_waypoint_reached()
        time.sleep(1)

    print(f"target coords: {target_coords}")

    if abs(target_coords[0])>30 or abs(target_coords[1])>30 or not (0<abs(target_coords[2])<30):
        print("target out of range")
        drone.land()
        return 
    
    print("moving to target")
    drone.set_destination(
       x=target_coords[0], y=target_coords[1], z=5, psi=0)
    while not drone.check_waypoint_reached():
        print("drone flying to target")
        time.sleep(1)
    print("target reached")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    # run control loop
    imaging_test_mission()
