import numpy as np
import cv2 as cv
from local_geolocation import GeoLocator

class DropZoneCoverage:
    def __init__(self, dropzone_local_coords, img_width, img_height, zone_width, zone_height, res_multiplier=2):
        dropzone_local_coords-=dropzone_local_coords[0]
        self.dropzone_coords = dropzone_local_coords
        bases_matrix = np.array([dropzone_local_coords[1], dropzone_local_coords[3]]).T
        transformation_matrix =  np.linalg.inv(bases_matrix)
        scale_matrix = res_multiplier*np.array([
            [zone_width, 0],
            [0, zone_height]
        ])
        self.world_to_dz_image =  scale_matrix @ transformation_matrix
        self.geolocator = GeoLocator()
        self.zone_img = np.zeros((zone_height*res_multiplier, zone_width*res_multiplier), dtype=np.uint8)
        self.img_width = img_width
        self.img_height = img_height

    def _get_coverage(self, location, angles):
        img_corners_world_coords = [
            self.geolocator.get_location(x,y,location,angles,(self.img_height,self.img_width)) 
            for x,y in [(0,0),(self.img_width,0),(self.img_width,self.img_height),(0,self.img_height)]
            ]
        img_dropzone_coords = [
            self.world_to_dz_image @ (np.array(coord[:2])-self.dropzone_coords[0]) for coord in img_corners_world_coords
        ]
        return img_dropzone_coords
    
    def add_coverage(self, location, angles):
        coverage_coords = self._get_coverage(location, angles)
        coverage_shape = np.zeros_like(self.zone_img)
        cv.fillConvexPoly(coverage_shape, np.array(coverage_coords, dtype=np.int32), 1)
        self.zone_img+=coverage_shape
        cv.imshow("zone", 50*self.zone_img)
        print(self.zone_img.shape, self.zone_img.max(), self.zone_img.dtype)
        cv.waitKey(0)
    
    def get_most_important_coverage_pts(self):
        raise NotImplementedError

if __name__=='__main__':
    dzc = DropZoneCoverage(
        dropzone_local_coords=np.array([(0,0), (300,0), (300,70), (0,70)]),
        img_width = 4000,
        img_height = 3000,
        zone_width=300,
        zone_height=70
    )

    dzc.add_coverage((150,35,50),(0,0,0))
    dzc.add_coverage((170,35,50),(0,0,0))
    dzc.add_coverage((150,10,23),(0,0,0))
    dzc.add_coverage((150,35,10),(0,0,0))