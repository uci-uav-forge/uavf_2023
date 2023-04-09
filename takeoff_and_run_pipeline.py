import rospy
import time
from navigation.guided_mission.py_gnc_functions import *
from imaging.pipeline import Pipeline

def imaging_test_mission():
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
        x=0, y=0, z=23, psi=0)
    
    while not drone.check_waypoint_reached():
        print('drone has not satisfied waypoint!')
        time.sleep(1)
        pass

    USE_GOPRO = True 
    imaging_pipeline = Pipeline(drone, (5568, 4176), dry_run=True, img_file="gopro" if USE_GOPRO else "imaging/gopro-image-5k.png", targets_file='imaging/targets.csv')
    start = time.perf_counter()
    imaging_pipeline.run(num_loops=10)
    end = time.perf_counter()
    print(imaging_pipeline.target_aggregator.targets)
    print(imaging_pipeline.target_aggregator.best_conf, imaging_pipeline.target_aggregator.target_gps)
    print(f"Time elapsed: {end - start}")

    drone.land()


if __name__ == '__main__':
    # initialize ROS node and get home position
    rospy.init_node("drone_GNC", anonymous=True)

    # run control loop
    imaging_test_mission()
