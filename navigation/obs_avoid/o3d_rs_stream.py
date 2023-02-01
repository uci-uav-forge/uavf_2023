import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import json 


def init_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


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


def o3d_stream(intr):
    pinhole_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    intr_tensor = o3d.core.Tensor(
        pinhole_intr.intrinsic_matrix)
    
    with open('o3d_rs_config.json') as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    cam = o3d.t.io.RealSenseSensor(sensor_config=rs_cfg)
    cam.start_capture(True)     # camera stream

    while(True):
        frame = cam.capture_frame(wait=True, align_depth_to_color=True)
        # Use frame's depth and intrinsic tensor to generate a point cloud
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            frame.depth, intr_tensor)
        o3d.visualization.draw_geometries([pcd.to_legacy()])


if __name__=='__main__':
    width = 424
    height = 240
    frame_rate = 15

    intr = get_intrinsic(width, height, frame_rate)
    print(intr)
