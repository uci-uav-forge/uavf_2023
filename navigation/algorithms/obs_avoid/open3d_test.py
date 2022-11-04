# sources:
# for downsampling: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Visualize-point-cloud
# for outlier removal: http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html


import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt


# visualization
dataset = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)
print(pcd)
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


# downsample
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


# color point cloud
print("Paint chair")
pcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])


# bounding volumes
st = time.time()
aabb = pcd.get_axis_aligned_bounding_box()
print('axis aligned bounding box runtime: ' + str(time.time()-st))
aabb.color = (1, 0, 0)
st = time.time()
obb = pcd.get_oriented_bounding_box()
print('oriented bounding box runtime: ' + str(time.time()-st))
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])


# radius outlier removal
print("Radius oulier removal")
cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.05)
display_inlier_outlier(down_pcd, ind)


# DBSCAN clustering
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])