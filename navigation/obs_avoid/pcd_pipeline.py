import open3d as o3d
import numpy as np
import time
import timeit
from numba import njit, prange


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


def process_pcd(pcd):
    '''Will downsample, filter, cluster, and segment a pointcloud. Returns an array of coordinates 
    for the centroid of each cluster as well as an array of dimensions for each bounding box.'''
    
    # downsample
    st = time.time()
    down_pcd = pcd.voxel_down_sample(voxel_size=60)
    # radius outlier removal
    fil_cl, ind = down_pcd.remove_radius_outlier(nb_points=40, radius=240)
    #fil_cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    '''
    # DBSCAN clustering
    labels = np.array(fil_cl.cluster_dbscan(eps=0.14, min_points=14, print_progress=False))

    
    # cluster segmentation
    N = labels.max() + 1
    centroids, box_dims, boxes = segment_clusters(N, fil_cl.points, labels)
    
    print(len(boxes))
    o3d.visualization.draw_geometries(boxes)
    '''
    return fil_cl
    return centroids, box_dims, fil_cl


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
