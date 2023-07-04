import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
from math import cos, sin, radians, degrees, tan
import os

import rospy
from std_msgs.msg import Int16MultiArray

from .rs_stream import post_process_filters, depth_to_pcd
from .pcd_pipeline import clean_pcd, yaw_rotation
from .threeD_obstacle_avoidance import obstacle_avoidance
from ..guided_mission.py_gnc_functions import gnc_api
os.chdir("navigation")

class RealsensePipeline:
    def __init__(self, res_width, res_height):
        # enable depth stream
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.depth, int(res_width), int(res_height), rs.format.z16, frame_rate
        )
        # start the pipeline
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(self.config)
        self.sensor = self.profile.get_device().first_depth_sensor()
        #max_range = sensor.set_option(sensor.set_option(rs.option.max_distance, 20))

        # initialize filters
        self.threshold = rs.threshold_filter(min_dist=0.01, max_dist=max_range)
        self.decimation = rs.decimation_filter(6)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.to_disparity = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
    
    def get_points(self):
        frames = self.pipe.wait_for_frames()
        
        #pitch, roll, yaw = 0, 0, 0

        st = time.time()
        depth_frame = post_process_filters(
            frames.get_depth_frame(), self.threshold, self.decimation, 
            self.spatial, self.temporal, self.hole_filling, self.to_disparity, self.to_depth
        )
        # generate point cloud
        o3d_pcd = depth_to_pcd(depth_frame)
        return o3d_pcd


def rs_stream(rs: RealsensePipeline):
    # drone api for attitude feedback, publisher to send waypoints
    drone = gnc_api()
    pcd_pub = rospy.Publisher(
        name="obs_avoid_pcd",
        data_class=Int16MultiArray,
        queue_size=1,
    )

    try: 
        # continuously run the depth stream
        while True:
            st = time.time()
            pitch, roll, yaw = drone.get_pitch_roll_yaw()    #pitch, roll, yaw in degrees
            o3d_pcd = rs.get_points()
            # get N x 3 array of detected points.
            detections = clean_pcd(o3d_pcd, pitch, roll) # this is in mm
            
            # convert from mm to m
            detections = detections / 1000 

            detections += np.array(drone.get_current_xyz())

            msg = Int16MultiArray()
            msg.data = list(np.round(detections.flatten()).astype(int))

            pcd_pub.publish(msg)
        
            print(time.time()-st)
            print()

    except KeyboardInterrupt:
        pipe.stop()


if __name__=='__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    
    width = 424
    height = 240
    frame_rate = 15
    
    max_range = 16 # m
    min_range = 4 # m, range when no safe path is found
    max_hdg = 43 # degrees, the angle of the FOV in 1 quadrant

    def gnc_api():
        from ..mock_drone import MockDrone
        return MockDrone(True)
    
    rospy.init_node("obstacle_detection_avoidance", anonymous=True)

    class MockRealsense:
        def get_points(self):
            vis_range = 4000
            n_pts = 5000
            time.sleep(0.05)
            pts = np.random.rand(n_pts,3)* 2*vis_range - vis_range + np.array([[0,0,vis_range]]*n_pts)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            return pcd



    rs_stream(MockRealsense())
    #rs_stream(RealsensePipeline(width,height))
