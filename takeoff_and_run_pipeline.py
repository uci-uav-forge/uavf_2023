import rospy
import time
from navigation.guided_mission.py_gnc_functions import *
from imaging.pipeline import Pipeline
import numpy as np
from threading import Thread
import sys

drone = gnc_api()

pipeline = Pipeline(
    localizer=drone,
    img_size=(5568, 4176),
    img_file="gopro",
    targets_file="test_mission_targets.csv",
    dry_run=False
)

target_coord = None


def run_pipeline():
    global target_coord
    print("start pipeline")
    try:
        pipeline.run(num_loops=1)
    except Exception as e:
        print(f"Encountered error while running imaging pipeline: {e}")
        return
    all_coords = pipeline.target_aggregator.get_target_coords()
    print(f"All coordinates: {all_coords}")
    target_coord = all_coords[0]
    print("done running pipeline")


def imaging_test_mission():
    # init drone api
    print("waiting to connect...")
    drone.wait4connect()
    print("drone connected")
    drone.set_mode_px4('OFFBOARD')
    print("offboard mode set")
    drone.initialize_local_frame()
    print("local frame initialized")

    print(f'Starting position: {drone.get_current_xyz()}')
    input("Press enter to launch drone")
    # print(f"Current pitch roll yaw: {drone.get_current_pitch_roll_yaw()}")
    drone.arm()
    drone.set_destination(x=0, y=0, z=23, psi=0)
    drone.set_mode_px4('OFFBOARD')
    print("destination set. taking off...")

    i = 0
    while not drone.check_waypoint_reached():
        print(f'drone flying to initial hover spot (checked {i} times)', end='\r')
        time.sleep(1)

    pipeline_thread = Thread(target=run_pipeline)
    print("\nStarting pipeline!")
    pipeline_thread.start()

    # this is necessary so that QGroundControl doesn't see a lack of input and enter failsafe mode to land early
    i=0
    while pipeline_thread.is_alive():
        print(f'waiting for pipeline to finish (checked {i} times)', end='\r')
        sys.stdout.flush()
        drone.check_waypoint_reached()
        time.sleep(1)

    print(f"target coords: {target_coord}")
    if target_coord is None:
        print("Target coordinate is None\nLanding...")
        drone.land() 
        return

    if np.linalg.norm(target_coord) > 30 or target_coord[2] < 0:
        print("target out of range")
        print(np.linalg.norm(target_coord))
        drone.land()
        return

    print("moving to target")
    drone.set_destination(x=target_coord[0], y=target_coord[1], z=5, psi=0)
    i=0
    while not drone.check_waypoint_reached():
        print(f'flying to target (checked {i} times)', end='\r')
        time.sleep(1)
    print("target reached")
    for i in range(5):
        print(f'flying to target (checked {i} times)', end='\r')
        drone.check_waypoint_reached()
        time.sleep(1)
    print("Landing after mission...")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)
    # run control loop
    imaging_test_mission()
