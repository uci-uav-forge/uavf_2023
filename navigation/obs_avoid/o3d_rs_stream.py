import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json 


class O3d_Visualizer():
    def __init__(self) -> None:
        self.source_pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        
        self.vis.create_window()
        #self.vis.add_geometry(self.source_pcd)
        self.threshold = 0.05
        self.save_image = False


    def update_vis(self, pcd) -> None:
        self.vis.remove_geometry(self.source_pcd)
        self.source_pcd = pcd.to_legacy()
        self.vis.add_geometry(self.source_pcd)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.save_image:
            self.vis.capture_screen_image("temp_%04d.jpg" % i)


# Get camera intrinsic 
def get_intrinsic(res_width, res_height, frame_rate):
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.depth, res_width, res_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(
        rs.stream.color, res_width, res_height, rs.format.bgr8, frame_rate)

    pipe = rs.pipeline()
    profile = pipe.start(rs_config)
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    return intr


def o3d_stream(intr, o3d_vis) -> None:    
    pinhole_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    
    intr_tensor = o3d.core.Tensor(pinhole_intr.intrinsic_matrix)
    
    with open('o3d_rs_cfg.json') as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    cam = o3d.t.io.RealSenseSensor()
    cam.init_sensor(sensor_config=rs_cfg, sensor_index=0)
    cam.start_capture(True)     # camera stream

    try: 
        while True:
            frame = cam.capture_frame(wait=True, align_depth_to_color=True)
            depth_pcd = o3d.t.geometry.PointCloud.create_from_depth_image(frame.depth, intr_tensor)
            o3d_vis.update_vis(depth_pcd)
    except KeyboardInterrupt:
        o3d_vis.vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


if __name__=='__main__':
    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    width = 1280
    height = 720
    frame_rate = 30

    intr = get_intrinsic(width, height, frame_rate)
    print(intr)
    
    o3d_vis = O3d_Visualizer()

    o3d_stream(intr, o3d_vis)
