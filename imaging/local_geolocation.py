import cameratransform as ct# pip install cameratransform


class GeoLocation:
    def __init__(self, img_size):
        self.img_size = img_size
    def get_location(self, image_x, image_y, location, angles):
        '''
            Returns the location of a pixel in the image in the world frame.
            location: assumed to be (x,y,z) where z is the height
            angles: assumed to be (pitch, roll, yaw)

            returns (x,y,z) in the world frame where z is the height.
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
        return cam.spaceFromImage([(image_x, image_y)])