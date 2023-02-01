# testing streaming intel D455 camera
import pyrealsense2 as rs
import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np

def main():
    # check if camera is connected to PC
    o3d.t.io.RealSenseSensor.list_devices()

    # connect to open3D
    rs = o3d.t.io.RealSenseSensor()
    #rs.init_sensor() # ignoring this function now, but important. It allows to specify streaming format, resolution, and fps
    # a json file has to be created, but that is easy. Also allows storing a video, but we don't care about that.
    rs.start_capture() # align_depth_to_color=false might make it faster
    pic = rs.capture_frame(True, True) # this streams a picture and depth, not just a point cloud
    print(type(pic))
    print(pic)
    #o3d.visualization.draw_geometries([pic])

    # show the two different aspects of the picture (demonstration only, will remove)
    show_in_matplotlib(pic)


    # convert to point cloud
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcl = convert_depth_frame_to_pointcloud(depth_image=pic.depth, camera_intrinsics=intrinsic)


    print('TYPES')
    print(type(intrinsic))
    print(type(pic.depth))
    #pcd = o3d.geometry.PointCloud.create_from_depth_image(pic.depth, intrinsic)
    o3d.geometry.create_point_cloud_from_depth_image(pic.depth, intrinsic)

    o3d.visualization.draw_geometries([pcd], zoom=0.5)



def show_in_matplotlib(picture: o3d.cpu.pybind.t.geometry.RGBDImage):
    plt.subplot(1, 2, 1)
    plt.title('Grayscale image')
    plt.imshow(picture.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(picture.depth)
    plt.show()

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

	z = depth_image.flatten() / 1000;
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z

if __name__ == '__main__':
    main()