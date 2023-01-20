# sources:
# for downsampling: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Visualize-point-cloud
# for outlier removal: http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html


import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt


def find_centroids(num_of_clusters: int, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    '''will return an array containing all centroids [[x1,y1,z1], [x2,y2,z2], ...]'''

    cluster_arr = np.empty((num_of_clusters, len(points), 3), dtype=float)
    sums = np.empty((num_of_clusters, 3), dtype=float)
    counter = np.empty(num_of_clusters, dtype=float)

    for i in range(len(points)):
        for j in range(num_of_clusters):
            if labels[i] == j:
                counter[j] += 1 # update counter
                cluster_arr[j][counter[j]] = points[i]
                sums[j] += points[i]
    
    # now calculate averages:
    print(cluster_arr)
    return sums / counter[:, None]


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
st = time.time()
downpcd = pcd.voxel_down_sample(voxel_size=0.1)
downsample_time = time.time() - st
print('downsampling runtime: ' + str(downsample_time))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

'''
# color point cloud
print("Paint chair")
pcd.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
'''


# radius outlier removal
print("Radius oulier removal")
st = time.time()
cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.2)
outlier_time = time.time() - st
print('outlier removal time' + str(outlier_time))
o3d.visualization.draw_geometries([cl],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
    

# DBSCAN clustering
st = time.time()
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        cl.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
cluster_time = time.time() - st
print("cluster computing time: " + str(cluster_time))


print(labels)
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([cl],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])



# bounding volumes
print('Bounding prism')
st = time.time()
aabb = cl.get_axis_aligned_bounding_box()
bounding_time = time.time() - st
print('axis aligned bounding box runtime: ' + str(bounding_time))
aabb.color = (1, 0, 0)
'''
st = time.time()
obb = pcd.get_oriented_bounding_box()
print('oriented bounding box runtime: ' + str(time.time()-st))
obb.color = (0, 1, 0)
'''
o3d.visualization.draw_geometries([cl, aabb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

print('total time: ' + str(downsample_time + outlier_time + bounding_time + cluster_time))