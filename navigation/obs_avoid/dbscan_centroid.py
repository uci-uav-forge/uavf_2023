# split a point cloud using DBSCAN clustering, then find each cluster's centroid
# helpful: http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html
# note: this code was written in python 3.10.7
# note2: sickit-learn also has dbscan, possibly compare speeds in future?

import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt

def main() -> None:
    '''everything happens here'''

    # load example point cloud:
    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)
    print(f'data loaded: {pcd}')

    # downsample
    vs = 0.1
    downpcd = pcd.voxel_down_sample(voxel_size=vs)
    print(f'point cloud downsampled by voxel size {vs}. Result: {downpcd}')

    # radius outlier removal
    cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.2)
    print(f'radius outlier removal applied. Resultl: {cl}')

    # DBSCAN clustering
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.asarray(
        cl.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))
    
    # find centroid of clusters (vip of this code)
    num_of_clusters = labels.max() + 1
    points = np.asarray(cl.points)
    st = time.time()
    centroids, cluster_arr = find_centroids(num_of_clusters, points, labels)
    print(cluster_arr)
    print('centroid calculation runtime: ' + str(time.time()-st))
    print(f'{num_of_clusters} found. centroids of clusters in [x,y,z] form:')
    print(centroids)

    # show point cloud, with clusters colored and their centroids
    # note: coloring may not work with if not 4 clusters, I haven't looked into it yet
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    visualize_pcd_and_centroids(pcd=cl, centroids=centroids)
    

def find_centroids(num_of_clusters: int, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    '''will return an array containing all centroids [[x1,y1,z1], [x2,y2,z2], ...]'''

    cluster_arr = np.zeros((num_of_clusters, len(points), 3), dtype=float)
    sums = np.zeros((num_of_clusters, 3), dtype=float)
    counter = np.zeros(num_of_clusters, dtype=int)

    for i in range(len(points)):
        for j in range(num_of_clusters):
            if labels[i] == j:
                counter[j] += 1 # update counter
                cluster_arr[j][counter[j]] = points[i]
                sums[j] += points[i]

    # now calculate averages:
    return sums/counter[:, None], cluster_arr 


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


def visualize_pcd_and_centroids(pcd: o3d.geometry.PointCloud, centroids: list) -> list:
    ''' take centroids, give back a mesh sphere at location of centroid'''
    things = [pcd]
    for i in range(len(centroids)):
        things.append(o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[centroids[i][0], centroids[i][1], centroids[i][2]]))
    
    o3d.visualization.draw_geometries(things)



if __name__ == '__main__':
    main()
