# sources:
# downsampling: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Visualize-point-cloud
# outlier removal: http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html


import open3d as o3d
import numpy as np
import time


def segment_clusters(num_cluster: int, points: np.ndarray, labels: np.ndarray) -> tuple:
    '''Will return an array containing all centroids [[x1,y1,z1], [x2,y2,z2], ..., [xN,yN,zN]]
       as well as a tuple containing N clusters.'''
    
    # initialize centroid array and cluster tuple
    centr_arr = np.zeros((num_cluster, 3), dtype=float)       
    pcd_tup = ()

    for i in range(num_cluster):
        # get indices of points belonging to ith cluster in array of all points
        indx_arr = np.where(labels == i)[0]
        cluster = np.zeros((indx_arr.size, 3), dtype=float)
        # insert the points at each of those indices into the cluster array
        for j in range(indx_arr.size):
            cluster[j] = points[indx_arr[j]]
        # convert cluster array to pcd and concatenate to tuple
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster)
        pcd_tup += (pcd,)
        centr_arr[i] = np.average(cluster, axis=0)
    
    return centr_arr, pcd_tup


if __name__ == '__main__':
    # visualization
    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)
    print(pcd)
    print()
    o3d.visualization.draw_geometries([pcd])

    # downsample
    print("Downsample the point cloud with a voxel of 0.05")
    st = time.time()
    downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    downsample_time = time.time() - st
    print('downsampling runtime: ' + str(downsample_time))
    print()
    o3d.visualization.draw_geometries([downpcd])

    # radius outlier removal
    st = time.time()
    fil_cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.2)
    outlier_time = time.time() - st
    print('outlier removal time: ' + str(outlier_time))
    print()
    o3d.visualization.draw_geometries([fil_cl])
        
    # DBSCAN clustering
    st = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(fil_cl.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
    cluster_time = time.time() - st
    print("cluster computing time: " + str(cluster_time))
    print()

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    fil_cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([fil_cl])

    # cluster segmentation
    st = time.time()
    N = labels.max() + 1
    centroids, clusters = segment_clusters(N, fil_cl.points, labels)
    segment_time = time.time() - st 
    print("segmentation time: " + str(segment_time))
    print()

    # bounding volumes
    st = time.time()
    bd_boxes = [pcd.get_axis_aligned_bounding_box() for pcd in clusters]
    bd_extents = [box.get_extent() for box in bd_boxes]
    bounding_time = time.time() - st
    print('axis aligned bounding box runtime: ' + str(bounding_time))
    print()

    for box in bd_boxes:
        box.color = (1, 0, 0)
    o3d.visualization.draw_geometries(bd_boxes + [fil_cl])
                                    
    print('total time: ' + str(downsample_time + outlier_time + bounding_time + cluster_time + segment_time))
    print()