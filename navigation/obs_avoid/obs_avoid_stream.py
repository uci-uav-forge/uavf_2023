from ..guided_mission.py_gnc_functions import gnc_api
from rs_stream import post_process_filters, rgbd_to_pcd, depth_to_pcd
from pcd_pipeline import process_pcd
from 3d_obstacle_avoidance import obstacle_avoidance

import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
from math import cos, sin, atan, radians
import rospy
from geometry_msgs.msg import Point


def yaw_rotation(raw_wp, yaw):
    theta = -yaw
    rad_theta = radians(theta)
    yaw_rot = np.array([
        [cos(rad_theta), -sin(rad_theta)],
        [sin(rad_theta), cos(rad_theta)]
    ])
    return yaw_rot @ raw_wp


# start the camera stream
def rs_stream(res_width, res_height, frame_rate):
    drone = gnc_api()
    avoid_pub = rospy.Publisher(
        name="obs_avoid_rel_coord",
        data_class=Point,
        queue_size=1,
    )

    config = rs.config()
    config.enable_stream(
        rs.stream.depth, int(res_width), int(res_height), rs.format.z16, frame_rate
    )

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
        while True:
            frames = pipe.wait_for_frames()
            pitch, roll, yaw = drone.get_pitch_roll_yaw()    #pitch, roll, yaw in degrees

            #st = time.time()
            depth_frame = post_process_filters(
                frames.get_depth_frame(), threshold, decimation, 
                spatial, temporal, hole_filling, to_disparity, to_depth
            )
            
            o3d_pcd = depth_to_pcd(depth_frame)
            
            centr_arr, box_arr, fil_cl = process_pcd(o3d_pcd, pitch, roll) # this is in mm
            if fil_cl == False: continue

            hdg = obstacle_avoidance(centr_arr, box_arr)
            raw_wp = np.array([max_dist * atan(hdg), max_dist])
            corrected_wp = yaw_rotation(raw_wp, yaw)
            
            rel_coord = Point()
            rel_coord.x = corrected_wp[0]
            rel_coord.y = corrected_wp[1]
            avoid_pub.publish(rel_coord)
            #print(time.time()-st)

    except KeyboardInterrupt:
        pipe.stop()


if __name__=='__main__':
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    width = 424
    height = 240
    frame_rate = 30
    max_range = 16 # m
    
    rs_stream(width, height, frame_rate, max_range)
