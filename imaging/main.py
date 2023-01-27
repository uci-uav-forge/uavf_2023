from fieldcapturer import FieldCapturer
from geolocator import Geolocator
from targetaggregator import TargetAggregator
from shapeInference.shape_inference import ShapeInference
from utils.target import Target

from ..navigation.guided_mission.run_mission import Localizer

import time
import cv2 as cv
import os 


class Pipeline:

    # Static variables
    VID_CAP_PORT = 1
    SLEEP_TIME = 10

    def __init__(self, 
    fieldCapturer: FieldCapturer, 
    geolocator: Geolocator, # Geolocation
    targetAggreg: TargetAggregator, # Remove duplicate detetection targets
    shapeInference: ShapeInference): # Shape model
        # self.detector = detector.Detector()
        self.field_capturer = fieldCapturer
        self.geolocator = geolocator
        self.target_aggregator = targetAggreg
        self.cam = cv.VideoCapture(self.VID_CAP_PORT)
        self.shapeInference = shapeInference
        self.localizer = Localizer()

    
    def getCurrentLocation(self):
        """
        Return the current local location of the UAV.
        """
        return self.localizer.get_current_location()


    def run(self):
        """
        Main run loop for the Imaging pipeline.
        """
        save_counter = 0
        while True:
            ret, img = self.cam.read()
            if not ret: raise Exception("Failed to grab frame")

            # save image to file
            path = 'savedGeoloc/images'
            img_name = "img{}.png".format(save_counter)
            cv.imwrite(os.path.join(path, img_name), img)

            # save location to file
            current_location = self.getCurrentLocation()
            f = open("savedGeoloc/locations.txt", "a")
            f.write("Save counter: {} | location: {}\n".format(save_counter, current_location))
            f.close()

            self.shapeInference.makePrediction(img)
            bounding_boxes = self.shapeInference.getBoundingBoxes() # Multiple bboxes per shape currently

            # Create target list
            targets = [Target(bounding_box=bbox) for bbox in bounding_boxes]

            for target in targets:
                bbox = target.bounding_box
                y_start, y_end = bbox[0], bbox[2]
                x_start, x_end = bbox[1], bbox[3]
                cropped_image = img[y_start:y_end, x_start:x_end]
                cv.imshow("Annotated image", cropped_image.astype(np.uint8))

            time.sleep(self.SLEEP_TIME) # loop

        