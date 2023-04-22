import time
from threading import Thread

import numpy as np

ACTUALLY_FLY_DRONE = False
USE_GOPRO = True

print(f"Actually flying drone: {ACTUALLY_FLY_DRONE}")
print(f"Using gopro: {USE_GOPRO}")

if ACTUALLY_FLY_DRONE:
    import rospy
    from navigation.guided_mission.py_gnc_functions import gnc_api
else:
    from navigation.mock_drone import MockDrone as gnc_api
from imaging.pipeline import Pipeline

drone = gnc_api()

pipeline = Pipeline(
    localizer=drone,
    img_size=(5568, 4176),
    img_file="gopro" if USE_GOPRO else "image0.png",
    targets_file="test_mission_targets.csv",
    dry_run=False
)

target_coord = None

def print_color(text, color, *args, **kwargs):
    colors_dict = {
        'red': '41',
        'green': '42',
        'yellow': '43',
        'blue': '44',
        'magenta': '45',
        'cyan': '46'
    }
    print(f'\x1b[0;30;{colors_dict[color]}m{text}\x1b[0m', *args, **kwargs)

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

    i = 1
    while not drone.check_waypoint_reached():
        print_color(f'drone flying to initial hover spot (checked {i} times)', color="magenta", end='\r')
        i+=1
        time.sleep(1)
    print()

    pipeline_thread = Thread(target=run_pipeline)
    print("Starting pipeline!")
    pipeline_thread.start()

    # this is necessary so that QGroundControl doesn't see a lack of input and enter failsafe mode to land early
    i = 1
    while pipeline_thread.is_alive():
        print_color(f'waiting for pipeline to finish (checked {i} times)', color="cyan",end='\r')
        i+=1
        drone.check_waypoint_reached()
        time.sleep(1)
    print()

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
    i = 1
    while not drone.check_waypoint_reached():
        print_color(f'flying to target (checked {i} times)', color = "green", end='\r')
        i+=1
        time.sleep(1)
    print()

    print("target reached")
    for i in range(5):
        print(f'hovering over target for {i} secs', end='\r')
        i+=1
        drone.check_waypoint_reached()
        time.sleep(1)
    print()
    print("Landing after mission...")
    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    if ACTUALLY_FLY_DRONE:
        rospy.init_node("drone_GNC", anonymous=True)
    # run control loop
    imaging_test_mission()
