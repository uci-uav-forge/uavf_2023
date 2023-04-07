import cameratransform as ct# pip install cameratransform


class GeoLocation:
    def __init__(self, img_size):
        self.img_size = img_size
    def get_location(self, image_x, image_y, location, angles):
        '''
            Returns the location of a pixel in the image in the world frame.
            location: assumed to be (x,z,y) where y is the height
            angles: assumed to be (heading, tilt, roll)
                    (-90,0,-90) is straight down with the top of the image up.

            returns (x,z,y) in the world frame where y is the height.
        '''
        heading, tilt, roll = angles
        x,z,y = location
        cam = ct.Camera(
            ct.RectilinearProjection(
                view_x_deg=67,
                view_y_deg=41,
                image=self.img_size),
            ct.SpatialOrientation(
                elevation_m=y,
                heading_deg=heading,
                tilt_deg=tilt,
                roll_deg=roll,
                pos_x_m=x,
                pos_y_m=z
            )
        )
        return cam.spaceFromImage([(image_x, image_y)])