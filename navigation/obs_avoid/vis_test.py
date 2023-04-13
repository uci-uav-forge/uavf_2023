import open3d as o3d
import numpy as np


class O3d_Visualizer():
    def __init__(self, source, target) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(source)
        self.vis.add_geometry(target)
        self.threshold = 0.05
        self.icp_iteration = 100
        self.save_image = False

    
    def update(self, source, target) -> None:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, self.threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        print(type(reg_p2l.transformation))
        source.transform(reg_p2l.transformation)
        self.vis.update_geometry(source)
        self.vis.poll_events()
        self.vis.update_renderer()
        if self.save_image:
            self.vis.capture_screen_image("temp_%04d.jpg" % i)
    

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd_data = o3d.data.DemoICPPointClouds()
    source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])

    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)
    
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)
    
    o3d_vis = O3d_Visualizer(source, target)

    for i in range(o3d_vis.icp_iteration):
        o3d_vis.update(source, target)
        
    o3d_vis.vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)