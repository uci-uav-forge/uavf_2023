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
    centroids = find_centroids(num_of_clusters, points, labels)
    print(f'{num_of_clusters} found. centroids of clusters in [x,y,z] form:')
    print(centroids)

    # show point cloud, with clusters colored and their centroids
    # note: coloring may not work with if not 4 clusters, I haven't looked into it yet
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    visualize_pcd_and_centroids(pcd=cl, centroids=centroids)
    



def find_centroids(num_of_clusters: int, points: np.array, labels: np.array) -> list:
    '''will return a list containing all centroids [[x,y,z], [x2,y2,z2],...]'''
    # tempting, but do not use: sums = [[]] * number. All elements will reference the same list!
    sums = [] # keeps track of sum of (x,y,z) of each cluster. first list keeps track of cluster 0, and so on...
    for i in range(num_of_clusters):
        sums.append([0,0,0]) # always: [x,y,z]
    counter = [0] * num_of_clusters

    for i in range(len(points)):
        for j in range(num_of_clusters):
            if labels[i] == j:
                sums[j][0] += points[i][0] # x
                sums[j][1] += points[i][1] # y
                sums[j][2] += points[i][2] # z
                counter[j] += 1 # update counter
    
    # now calculate averages:
    centroids = []
    for i in range(num_of_clusters):
        centroids.append([None, None, None])
        centroids[i][0] = sums[i][0]/counter[i]
        centroids[i][1] = sums[i][1]/counter[i]
        centroids[i][2] = sums[i][2]/counter[i]
    
    return centroids

def visualize_pcd_and_centroids(pcd: o3d.geometry.PointCloud, centroids: list) -> list:
    ''' take centroids, give back a mesh sphere at location of centroid'''
    things = [pcd]
    for i in range(len(centroids)):
        things.append(o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[centroids[i][0], centroids[i][1], centroids[i][2]]))
    
    o3d.visualization.draw_geometries(things)



if __name__ == '__main__':
    main()
