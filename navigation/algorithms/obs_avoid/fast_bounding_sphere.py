# Next step: prune elements within point cloud

from typing import Tuple
import numpy as np
import pcl
import time 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def downsample(cloud):
    fil = cloud.make_voxel_grid_filter()
    fil.set_leaf_size(0.5 , 0.5, 0.5)
    down_cloud = fil.filter()

    return down_cloud


def filter_outliers(cloud):
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(10)
    fil.set_std_dev_mul_thresh(10E-10)
    fil_cloud = fil.filter()
    fil_data = np.asarray(fil_cloud)
    
    return fil_data


def bounding_sphere_naive(positions: np.ndarray) -> Tuple[np.ndarray, float]:
    bbox = np.vstack([np.amin(positions, 0), np.amax(positions, 0)])
    center = np.average(bbox, axis=0)
    radius = np.linalg.norm(center - positions, axis=1).max()
    
    return center, radius


def gen_plot(cloud):
    #print(pt_cld)
    cloudX = cloud[:, 0]   
    cloudY = cloud[:, 1]
    cloudZ = cloud[:, 2]

    return np.array([cloudX, cloudY, cloudZ])


def param_sphere(center, radius):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = center[0] + radius*np.cos(u)*np.sin(v)
    y = center[1] + radius*np.sin(u)*np.sin(v)
    z = center[2] + radius*np.cos(v)
    
    return np.array([x, y, z])


def draw_plot(pt_cloud, sphere):
    fig = plt.figure(figsize=(9,10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(pt_cloud[0], pt_cloud[1], pt_cloud[2], c='red', alpha=1)
    ax.scatter(sphere[0], sphere[1], sphere[2], c='blue', alpha=0.01)

    plt.show()


if __name__ == '__main__':
    # control group
    n = 100000
    data = np.random.randn(n, 3)
    center, r = bounding_sphere_naive(positions=data)
    sphere = param_sphere(center, r)
    data_plot = gen_plot(data)
    draw_plot(data_plot, sphere)

    # add outliers and convert to point cloud
    data = np.append(data, 100*np.random.randn(500, 3), axis=0)
    float_data = data.astype(np.float32)
    cloud = pcl.PointCloud()
    cloud.from_array(float_data)

    # downsample and filter point cloud, calculate sphere
    st = time.time()
    down_cloud = downsample(cloud)
    fil_data = filter_outliers(down_cloud)
    center, r = bounding_sphere_naive(positions=fil_data)
    print('filtered execution time: ' + str(time.time() - st))
    data_plot = gen_plot(fil_data)

    # rerun with original point cloud
    st = time.time()
    data = np.asarray(cloud)
    center1, r1 = bounding_sphere_naive(positions=data)
    print('unfiltered execution time: ' + str(time.time() - st))
    data_plot1 = gen_plot(data)

    #compare filtered and unfiltered clouds and spheres
    sphere = param_sphere(center, r)
    sphere1 = param_sphere(center1, r1)
    draw_plot(data_plot, sphere)
    draw_plot(data_plot1, sphere1)
