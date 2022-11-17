from typing import Tuple
import numpy as np
import pcl
import time 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def downsample(cloud):
    vg = cloud.make_voxel_grid_filter()
    vg.set_leaf_size(0.5 , 0.5, 0.5)
    down_cloud = vg.filter()
    return down_cloud


def filter_outliers(cloud):
    fil = cloud.make_statistical_outlier_filter()
    fil.set_mean_k(5)
    fil.set_std_dev_mul_thresh(10E-10)
    fil_cloud = fil.filter()
    fil_data = np.asarray(fil_cloud)
    return fil_data


def bound_sphere(positions: np.ndarray) -> Tuple[np.ndarray, float]:
    bbox = np.vstack([np.amin(positions, 0), np.amax(positions, 0)])
    center = np.average(bbox, axis=0)
    radius = np.linalg.norm(center - positions, axis=1).max()
    return center, radius


def bound_box(positions: np.ndarray) -> np.ndarray:
    p0 = np.amin(positions, 0)
    p1 = np.amax(positions, 0)
    p2 = np.array([p1[0], p0[1], p0[2]])
    p3 = np.array([p0[0], p1[1], p0[2]])
    p4 = np.array([p1[0], p1[1], p0[2]])
    p5 = np.array([p0[0], p0[1], p1[2]])
    p6 = np.array([p1[0], p0[1], p1[2]])
    p7 = np.array([p0[0], p1[1], p1[2]])
    return np.array([p0, p1, p2, p3, p4, p5, p6, p7])


def gen_plot(data):
    #print(pt_cld)
    cloudX = data[:, 0]   
    cloudY = data[:, 1]
    cloudZ = data[:, 2]
    return np.array([cloudX, cloudY, cloudZ])


def param_sphere(center, radius):
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
    x = center[0] + radius*np.cos(u)*np.sin(v)
    y = center[1] + radius*np.sin(u)*np.sin(v)
    z = center[2] + radius*np.cos(v)
    return np.array([x, y, z])


def draw_plot(data_plot, shape, check_cube):
    fig = plt.figure(figsize=(9,10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data_plot[0], data_plot[1], data_plot[2], c='red', alpha=1)
    if (check_cube):
        ax.scatter(shape[0], shape[1], shape[2], c='blue', alpha=1)
    else:
        ax.scatter(shape[0], shape[1], shape[2], c='blue', alpha=0.01)
    plt.show()


if __name__ == '__main__':
    # control group
    n = 100000
    data = np.random.randn(n, 3)

    # add outliers and convert to point cloud
    data = np.append(data, 100*np.random.randn(500, 3), axis=0)
    float_data = data.astype(np.float32)
    cloud = pcl.PointCloud()
    cloud.from_array(float_data)

    # downsample and filter point cloud, calculate sphere
    st = time.time()
    down_cloud = downsample(cloud)
    fil_data = filter_outliers(down_cloud)
    cube_data = bound_box(positions=fil_data)
    #center, r = bound_sphere(positions=fil_data)
    print('filtered execution time: ' + str(time.time() - st))
    data_plot = gen_plot(fil_data)

    # rerun with original point cloud
    st = time.time()
    data = np.asarray(cloud)
    cube_data1 = bound_box(positions=data)
    #center1, r1 = bound_sphere(positions=data)
    print('unfiltered execution time: ' + str(time.time() - st))
    data_plot1 = gen_plot(data)

    #compare filtered and unfiltered clouds and cubes
    #sphere = param_sphere(center, r)
    #sphere1 = param_sphere(center1, r1)
    cube_plot = gen_plot(cube_data)
    cube_plot1 = gen_plot(cube_data1)
    draw_plot(data_plot, cube_plot, True)
    draw_plot(data_plot1, cube_plot1, True)
