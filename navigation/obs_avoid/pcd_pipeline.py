import open3d as o3d
import numpy as np
import time
import timeit
from numba import njit, prange
from math import radians, cos, sin


def segment_clusters(num_cluster: int, points: np.ndarray, labels: np.ndarray) -> tuple:
    '''Will return an array containing all centroids [[x1,y1,z1], [x2,y2,z2], ..., [xN,yN,zN]]
       as well as an array containing the dimensions of each bounding box'''
    
    # initialize centroid and bounding box dimension array
    centr_arr = np.zeros((num_cluster, 3), dtype=float)       
    box_arr = np.zeros((num_cluster, 3), dtype=float)
    box_list = []

    for i in range(num_cluster):
        # get indices of points belonging to ith cluster in array of all points
        indx_arr = np.where(labels == i)[0]
        cluster = np.zeros((indx_arr.size, 3), dtype=float)
        # insert the points at each of those indices into the cluster array
        for j in range(indx_arr.size):
            cluster[j] = points[indx_arr[j]]

        # convert cluster array to pcd and get each bounding box's dimensions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster)
        bd_box = pcd.get_axis_aligned_bounding_box()
        box_dim = bd_box.get_extent()

        centr_arr[i] = np.average(cluster, axis=0)
        box_arr[i] = box_dim
        box_list.append(bd_box)
    
    return centr_arr, box_arr, box_list


def apply_rotations(centroids, box_dims, pitch, roll):
    # rotation from realsense coords to standard attitude coord frame
    std_rot = np.array([
        [0, 0, -1],
        [1, 0, 0],
        [0 ,-1, 0]
    ])
    std_centroids = centroids @ std_rot.T
    std_box_dims = box_dims @ std_rot.T

    rad_pitch = radians(pitch)
    rad_roll = radians(roll)    

    # correction rotation due to drone attitude
    pitch_rot = np.array([
        [cos(rad_pitch), 0, sin(rad_pitch)],
        [0, 1, 0],
        [-sin(rad_pitch), 0,    cos(rad_pitch)]
    ])
    roll_rot = np.array([
        [1, 0, 0],
        [0, cos(rad_roll), -sin(rad_roll)],
        [0, sin(rad_roll), cos(rad_roll)]
    ])
    tilt_centroids = std_centroids @ pitch_rot.T @ roll_rot.T

    # rotation from standard attitude coord frame to obstacle avoidance coord frame
    avoidance_rot = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    centr_arr = tilt_centroids @ avoidance_rot.T
    box_arr = std_box_dims @ avoidance_rot.T

    return centr_arr, box_arr


def yaw_rotation(raw_wp, yaw):
    theta = -yaw
    rad_theta = radians(theta)
    yaw_rot = np.array([
        [cos(rad_theta), -sin(rad_theta)],
        [sin(rad_theta), cos(rad_theta)]
    ])
    return yaw_rot @ raw_wp


def process_pcd(pcd, pitch, roll):
    '''Will downsample, filter, cluster, and segment a pointcloud. Returns an array of coordinates 
    for the centroid of each cluster as well as an array of dimensions for each bounding box.'''
    
    # downsample
    down_pcd = pcd.voxel_down_sample(voxel_size=600)#200
    # radius outlier removal
    fil_cl, ind = down_pcd.remove_radius_outlier(nb_points=24, radius=2100)#15,400
    #fil_cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # DBSCAN clustering
    labels = np.array(fil_cl.cluster_dbscan(eps=1200, min_points=12, print_progress=False))

    # cluster segmentation
    try:
        N = labels.max() + 1
    except ValueError:
        return False, False, False

    centroids, box_dims, boxes = segment_clusters(N, fil_cl.points, labels)
    centr_arr, box_arr = apply_rotations(centroids, box_dims, pitch, roll)

    #print(centroids)
    #print(centr_arr)
    #print(np.asarray(fil_cl.points))
    #print(boxes)
    #o3d.visualization.draw_geometries(boxes, zoom=0.5)
    return centr_arr, box_arr, fil_cl


if __name__ == '__main__':
    # testing runtime with timeit
    SETUP_CODE = '''
import open3d as o3d
import numpy as np
from __main__ import process_pcd
dataset = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)'''

    TEST_CODE = '''
centroids, dims, fil_cl = process_pcd(pcd)'''

    times = timeit.repeat(setup = SETUP_CODE, stmt = TEST_CODE, repeat=1000, number=1)
    print('Best case pcd processing time: {}'.format(np.min(times)))
    print('Average pcd processing time: {}'.format(np.average(times)))

    
    '''# testing runtime with time.time()
    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

    st = time.time()
    centroids, dims, fil_cl = process_pcd(pcd)
    et = time.time() - st
    
    print()
    print('centroids: ' + str(centroids))
    print('dimensions of bounding boxes: ' + str(dims))
    print('pcd pipeline runtime ' + str(et))
    '''
