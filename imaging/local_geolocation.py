import numpy as np
import cameratransform as ct  # pip install cameratransform


class GeoLocator:
    def __init__(self):
        pass

    def get_location(self, image_x: int, image_y: int, location:"tuple[float,float,float]", angles:"tuple[float,float,float]", img_size: "tuple[int,int]") -> "tuple[float,float,float]":
        """
        img_size should be (height, width)

        Returns the location of a pixel in the image in the world frame.
        location: assumed to be (x,y,z) where z is the height
        angles: assumed to be (heading, tilt, roll)

        returns (x,y,z) in the world frame where z is the height.

        it assumes image plane is at z=0, the angle orientation at (0,0,0) corresponds to looking straight down
        with the drone pointed in the positive y direction, and the rotations are applied in this order:
        (yaw, pitch, roll) (rotation around z-axis, then x-axis, then z-axis again)
        https://cameratransform.readthedocs.io/en/latest/coordinate_systems.html#space
        """
        heading, tilt, roll = angles
        x, y, z = location
        cam = ct.Camera(
            #https://community.gopro.com/s/article/HERO10-Black-Digital-Lenses-FOV-Informations?language=en_US
            #we're on 4:3 ratio narrow mode no hypersmooth 
            ct.RectilinearProjection(                
                view_x_deg=73,
                view_y_deg=58,
                image_width_px=img_size[1],
                image_height_px=img_size[0]),
            ct.SpatialOrientation(
                elevation_m=z,
                heading_deg=heading,
                tilt_deg=tilt,
                roll_deg=roll,
                pos_x_m=x,
                pos_y_m=y
            )
        )
        camera_world_coords = cam.spaceFromImage([(image_x, image_y)])[0]
        # in the drone's reference frame, the top of the image is positive x and the right side is negative y
        offset_from_center = camera_world_coords - np.array(location)
        return location[0] + offset_from_center[1], location[1] - offset_from_center[0], 0


if __name__ == "__main__":
    geolocator = GeoLocator((5568, 4176))
    loc = geolocator.get_location(0, 0, (5, 5, 10), (90, 0, 0))
    print(loc, np.linalg.norm(loc), loc[2] > 0, type(loc))
