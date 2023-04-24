import numpy as np
import cv2 as cv
from .local_geolocation import GeoLocator

class ZoneCoverageTracker:
    def __init__(self, dropzone_local_coords: np.ndarray):
        PIXELS_PER_METER = 3
        dropzone_local_coords-=dropzone_local_coords[0]
        self.dropzone_coords = dropzone_local_coords
        bases_matrix = np.array([dropzone_local_coords[1], dropzone_local_coords[3]]).T
        transformation_matrix =  np.linalg.inv(bases_matrix)
        l1 = np.linalg.norm(dropzone_local_coords[0]-dropzone_local_coords[1])
        l2 = np.linalg.norm(dropzone_local_coords[0]-dropzone_local_coords[3])
        zone_width = np.ceil(l1*PIXELS_PER_METER).astype(np.int32)
        zone_height = np.ceil(l2*PIXELS_PER_METER).astype(np.int32)
        scale_matrix = np.array([
            [zone_width, 0],
            [0, zone_height]
        ])
        self.world_to_dz_image =  scale_matrix @ transformation_matrix
        self.geolocator = GeoLocator()
        self.zone_img = np.zeros((zone_height, zone_width), dtype=np.uint8)

    def _get_coverage(self, location, angles, img_dims):
        h, w = img_dims 
        img_corners_world_coords = [
            self.geolocator.get_location(x,y,location,angles,(h,w)) 
            for x,y in [(0,0),(w,0),(w,h),(0,h)]
            ]
        img_dropzone_coords = [
            self.world_to_dz_image @ (np.array(coord[:2])-self.dropzone_coords[0]) for coord in img_corners_world_coords
        ]
        return img_dropzone_coords
    
    def add_coverage(self, location, angles, img_dims=(3, 4)):
        coverage_coords = self._get_coverage(location, angles, img_dims)
        coverage_shape = np.zeros_like(self.zone_img)
        cv.fillConvexPoly(coverage_shape, np.array(coverage_coords, dtype=np.int32), 1)
        self.zone_img+=coverage_shape

    def get_coverage_image(self):
        return self.zone_img * (255//self.zone_img.max()) if self.zone_img.max()>0 else self.zone_img

    def get_most_important_coverage_pts(self):
        raise NotImplementedError

if __name__=='__main__':
    dzc = ZoneCoverageTracker(
        dropzone_local_coords=np.array([(0,0), (300,0), (300,70), (0,70)]),
    )

    dzc.add_coverage((150,35,75),(0,0,0))
    dzc.add_coverage((100,30,30),(0,0,0))

    cv.imshow('coverage',dzc.get_coverage_image())
    cv.waitKey(0)