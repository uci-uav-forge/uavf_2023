import pyrealsense2 as rs
import numpy as np
import time
from math import cos, sin, atan, radians
import os

import rospy
from geometry_msgs.msg import Point

from .rs_stream import post_process_filters, depth_to_pcd
from .pcd_pipeline import process_pcd, yaw_rotation
from .threeD_obstacle_avoidance import obstacle_avoidance
from ..guided_mission.py_gnc_functions import gnc_api
os.chdir("navigation")


def rs_stream(res_width, res_height, frame_rate, max_range, avoid_range):
    # drone api for attitude feedback, publisher to send waypoints
    drone = gnc_api()
    avoid_pub = rospy.Publisher(
        name="obs_avoid_rel_coord",
        data_class=Point,
        queue_size=1,
    )

    # enable depth stream
    config = rs.config()
    config.enable_stream(
        rs.stream.depth, int(res_width), int(res_height), rs.format.z16, frame_rate
    )

    # start the pipeline
    pipe = rs.pipeline()
    profile = pipe.start(config)
    sensor = profile.get_device().first_depth_sensor()
    #max_range = sensor.set_option(sensor.set_option(rs.option.max_distance, 20))

    # initialize filters
    threshold = rs.threshold_filter(min_dist=0.01, max_dist=max_range)
    decimation = rs.decimation_filter(6)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    to_disparity = rs.disparity_transform(True)
    to_depth = rs.disparity_transform(False)

    try: 
        # continuously run the depth stream
        while True:
            frames = pipe.wait_for_frames()
            pitch, roll, yaw = drone.get_pitch_roll_yaw()    #pitch, roll, yaw in degrees
            #pitch, roll, yaw = 0, 0, 0

            st = time.time()
            depth_frame = post_process_filters(
                frames.get_depth_frame(), threshold, decimation, 
                spatial, temporal, hole_filling, to_disparity, to_depth
            )
            
            # generate point cloud
            o3d_pcd = depth_to_pcd(depth_frame)
            
            # get N x 3 arrays of object centroids and their bounding volume dimensions
            centr_arr, box_arr, fil_cl = process_pcd(o3d_pcd, pitch, roll) # this is in mm
            if fil_cl == False: continue
            
            # convert from mm to m
            centr_arr = centr_arr / 1000 
            box_arr = box_arr / 1000

            # obstacle avoidance returns heading angle within FOV -> waypoint at edge of FOV
            # rotate to waypoint corresponding to yaw to get relative coordinates in local frame
            hdg = obstacle_avoidance(centr_arr, box_arr)
            if hdg:
                raw_wp = np.array([avoid_range * atan(hdg), avoid_range])
                corrected_wp = yaw_rotation(raw_wp, yaw)

                print(hdg)
                print(corrected_wp)
                
                # create and publish point message
                rel_coord = Point()
                rel_coord.x = corrected_wp[0]
                rel_coord.y = corrected_wp[1]
                avoid_pub.publish(rel_coord)
            
            print(time.time()-st)
            print()

    except KeyboardInterrupt:
        pipe.stop()


if __name__=='__main__':
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    width = 424
    height = 240
    frame_rate = 15
    max_range = 16 # m
    avoid_range = 16 # m     the range at which the avoidance waypoint is published
    
    rospy.init_node("obstacle_detection_avoidance", anonymous=True)
    rs_stream(width, height, frame_rate, max_range, avoid_range)
