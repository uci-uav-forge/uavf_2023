import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json 
from .pcd_pipeline import process_pcd
from numba import njit, prange
import time
import math


class O3d_Visualizer():
    def __init__(self) -> None:
        self.source_pcd = o3d.geometry.PointCloud()
        self.source_img = o3d.geometry.Image()
        self.vis = o3d.visualization.Visualizer()
        
        self.vis.create_window()
        self.vis.add_geometry(self.source_pcd)
        self.vis.add_geometry(self.source_img)


    def update_pcd(self, pcd) -> None:
        self.vis.remove_geometry(self.source_pcd)
        self.source_pcd = pcd
        self.vis.add_geometry(self.source_pcd)
        
        self.vis.poll_events()
        self.vis.update_renderer()
    

    def update_img(self, img) -> None:
        self.vis.remove_geometry(self.source_img)
        self.source_img  = img
        self.vis.add_geometry(self.source_img)
        
        self.vis.poll_events()
        self.vis.update_renderer()


def post_process_filters(input_frame, threshold, decimation, spatial, temporal, hole_filling, to_disparity, to_depth):
    frame = threshold.process(input_frame)
    #frame = decimation.process(input_frame)
    
    #frame = to_disparity.process(input_frame)
    #frame = spatial.process(frame)
    
    #frame = to_depth.process(frame)
    #frame = hole_filling.process(frame)
    return frame


def rgbd_to_pcd(depth_frame, color_frame, intr):
    depth_arr = np.asanyarray(depth_frame.get_data())
    color_arr = np.asanyarray(color_frame.get_data())

    depth_img = o3d.geometry.Image(depth_arr.astype(np.float32))
    color_img = o3d.geometry.Image(color_arr.astype(np.uint8))
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intr)
    return pcd


def depth_to_pcd(depth_frame):
    prof = depth_frame.get_profile()
    intr = prof.as_video_stream_profile().get_intrinsics()
    pinhole_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )

    depth_arr = np.asanyarray(depth_frame.get_data())
    depth_img = o3d.geometry.Image(depth_arr.astype(np.float32))
    
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img, pinhole_intr)
    pcd.transform([[1, 0, 0, 0], 
                   [0, -1, 0, 0], 
                   [0, 0, -1, 0], 
                   [0, 0,  0,  1]])
    return pcd


# start the camera stream
def test_rs_stream(res_width, res_height, frame_rate, o3d_vis):
    #drone = gnc_api()

    config = rs.config()
    config.enable_stream(
        rs.stream.depth, int(res_width), int(res_height), rs.format.z16, frame_rate
    )
    '''
    config.enable_stream(
        rs.stream.color, int(res_width), int(res_height), rs.format.rgb8, frame_rate
    )'''

    pipe = rs.pipeline()
    profile = pipe.start(config)
    sensor = profile.get_device().first_depth_sensor()
    #max_range = sensor.set_option(sensor.set_option(rs.option.max_distance, 20))

    # initialize filters
    #align = rs.align(rs.stream.depth)
    threshold = rs.threshold_filter(min_dist=0.01, max_dist=16.0)
    decimation = rs.decimation_filter(6)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    to_disparity = rs.disparity_transform(True)
    to_depth = rs.disparity_transform(False)

    try: 
        while True:
            frames = pipe.wait_for_frames()
            pitch, roll = 0, 0

            st = time.time()
            depth_frame = post_process_filters(
                frames.get_depth_frame(), 
                threshold, decimation, spatial, temporal, hole_filling,
                to_disparity, to_depth
            )
            
            o3d_pcd = depth_to_pcd(depth_frame)   
            '''
            crop_pcd = o3d.geometry.crop_point_cloud(o3d_pcd, 
                min_bound=np.array([-math.inf, -math.inf, -math.inf]), 
                max_bound=np.array([math.inf, math.inf, math.inf])
            )'''
            
            centr_arr, box_arr, fil_cl = process_pcd(o3d_pcd, pitch, roll)
            print(time.time()-st)

            if fil_cl == False: pass
            else: 
                o3d_vis.update_pcd(fil_cl)
                #print(np.asarray(fil_cl.points))
            #o3d_vis.update_img(depth_img)

    except KeyboardInterrupt:
        pipe.stop()
        o3d_vis.vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


if __name__=='__main__':
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    width = 424
    height = 240
    frame_rate = 15

    o3d_vis = O3d_Visualizer()

    test_rs_stream(width, height, frame_rate, o3d_vis)
