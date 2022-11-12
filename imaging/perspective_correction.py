import cameratransform as ct
import matplotlib.pyplot as plt
# display a top view of the image
im = plt.imread("data-gen/grid-roll-23-deg.png")
# intrinsic camera parameters
image_size = (5312, 2988)    # in px. Needs to match the input image size.

# initialize the camera
# test_values = [0,20,40,60]
# for i, val in enumerate(test_values):
cam = ct.Camera(
    ct.RectilinearProjection(
        view_x_deg=67, 
        view_y_deg=41,
        image=image_size),
    ct.SpatialOrientation(
        elevation_m=75,
        heading_deg=-90,
        tilt_deg=40, # not sure why we need to overshoot this to get the correction to work.
        roll_deg=-90,
        pos_x_m=70 # also not sure why this shouldn't just be zero
    )
)

top_im = cam.getTopViewOfImage(im, extent = [-50, 50, -28, 28])
# plt.subplot(100+10*len(test_values)+i+1)
plt.subplot(211)
plt.title("Original Image")
plt.imshow(im)
plt.subplot(212)
plt.title("Corrected Image")
plt.imshow(top_im)
plt.show()