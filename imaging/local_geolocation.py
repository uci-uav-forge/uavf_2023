import numpy as np
import cameratransform as ct# pip install cameratransform


class GeoLocation:
    def __init__(self, img_size):
        self.img_size = img_size
    def get_location(self, image_x, image_y, location, angles) -> np.ndarray:
        '''
            Returns the location of a pixel in the image in the world frame.
            location: assumed to be (x,y,z) where z is the height
            angles: assumed to be (pitch, roll, yaw)

            returns (x,y,z) in the world frame where z is the height.

            it assumes image plane is at z=0, the angle orientation at (0,0,0) corresponds to looking straight down with the drone pointed in the positive y direction, and the rotations are applied in this order: (yaw, pitch, roll) (rotation around z axis, then x axis, then z axis again)
        '''
        pitch, roll, yaw = angles
        x,y,z = location
        cam = ct.Camera(
            ct.RectilinearProjection(
                view_x_deg=67,
                view_y_deg=41,
                image=self.img_size),
            ct.SpatialOrientation(
                elevation_m=z,
                heading_deg=yaw,
                tilt_deg=pitch,
                roll_deg=roll,
                pos_x_m=x,
                pos_y_m=y
            )
        )
        return cam.spaceFromImage([(image_x, image_y)])[0]
    
if __name__=="__main__":
    geolocator = GeoLocation((1000, 1000))
    loc = geolocator.get_location(500,300,(0,0,0),(0,0,0))
    print(loc, np.linalg.norm(loc), loc[2]>0, type(loc))