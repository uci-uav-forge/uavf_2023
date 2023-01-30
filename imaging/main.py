# from fieldcapturer import FieldCapturer
# from geolocator import Geolocator
# from targetaggregator import TargetAggregator
# from shapeInference.shape_inference import ShapeInference
# from utils.target import Target

import sys
sys.path.append("..")
# from ..navigation.guided_mission.run_mission import Localizer
import letter_detection.LetterDetector as letter_detection
import shape_detection.src.model as shape_model
import torch
import time
import cv2 as cv
import os
import numpy as np
from PIL import Image
import tensorflow as tf

class Pipeline:

    # Static variables
    VID_CAP_PORT = 1
    SLEEP_TIME = 10

    """
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
    """

    def __init__(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus: # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

        # self.localizer = Localizer()
        self.tile_resolution=512# has to match img_size of the model, which is determined by which one we use.
        backbone_name="efficientnet_b0"
        model = shape_model.EfficientDetModel(
            num_classes=13,
            img_size=512,
            model_architecture=backbone_name # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
            )
        model_file=f"shape_detection/src/efficientnet_b0_pytorch_25epoch.pt"
        model.load_state_dict(torch.load(model_file))
        model.to("cuda")
        model.eval()
        self.shape_model = model

        self.letter_detector = letter_detection.LetterDetector("trained_model.h5")


    
    def getCurrentLocation(self):
        """
        Return the current local location of the UAV.
        """
        return self.localizer.get_current_location()

    
    def logGeolocation(self, counter: int, img, loc):
        """
        Save image and corresponding location in the savedGeoloc directory. 
        The image number corresponds to the save counter in savedGeoloc/locations.txt
        """
        # save image
        img_name = "img{}.png".format(counter)
        cv.imwrite(os.path.join('savedGeoloc/images', img_name), img)

        # save location
        f = open("savedGeoloc/locations.txt", "a")
        f.write("Save counter: {} | location: {}\n".format(counter, loc))
        f.close()

    def loop(self):
        # ret, img = self.cam.read()
        # if not ret: raise Exception("Failed to grab frame")
        # current_location = self.getCurrentLocation()
        # self.logGeolocation(save_counter, img, current_location)
        img = cv.imread("gopro-image-5k.png")
        h,w = img.shape[:2]
        n_horizontal_tiles = w//self.tile_resolution
        n_vertical_tiles = h//self.tile_resolution
        all_tiles = []
        h_tiles = np.split(img,range(self.tile_resolution,(n_horizontal_tiles+1)*self.tile_resolution,self.tile_resolution),axis=1)
        tile_offsets_x_y: 'list[tuple]'  = []

        for i,h_tile in enumerate(h_tiles):
            y_offset = i*self.tile_resolution
            v_tiles = np.split(h_tile,range(self.tile_resolution,(n_vertical_tiles+1)*self.tile_resolution,self.tile_resolution),axis=0)
            for j,tile in enumerate(v_tiles):
                all_tiles.append(tile)
                tile_offsets_x_y.append((j*self.tile_resolution,y_offset))

        pil_images = [Image.fromarray(tile) for tile in all_tiles]
        batch_size = 1

        bboxes, labels, confidences = [], [], [] # bboxes are xyxy
        # `map(list,...` makes sure the correct `predict` type overload (List[PIL.Image]) is being called since it tries using an identity function for ndarray[PIL.Image]
        for batch in map(list,np.split(
            ary=pil_images, 
            indices_or_sections=range(batch_size, len(pil_images),batch_size),
            axis=0)
        ):
            res = self.shape_model.predict(batch)
            b,l,c=res
            bboxes.extend(b)
            labels.extend(l)
            confidences.extend(l)
        print(tile_offsets_x_y[:5])
        for tile_index in range(len(bboxes)):
            if len(bboxes[tile_index])>0:
                print(tile_index, bboxes[tile_index])
                label = labels[tile_index]
                x_offset, y_offset = tile_offsets_x_y[tile_index]
                cropped_image = cv.cvtColor(
                    src=np.array(pil_images[tile_index]),
                    code=cv.COLOR_RGB2GRAY
                )
                just_letter_images=[]
                for box_x0, box_y0, box_x1, box_y1 in bboxes[tile_index]:
                    box_w=int(box_x1)-int(box_x0)
                    box_h=int(box_y1)-int(box_y0)
                    box_crop=cropped_image[
                        int(box_x0):int(box_x1),
                        int(box_y0):int(box_y1)
                        ]
                    print(box_w, box_h, box_crop.shape)
                    just_letter_images.append(
                        np.pad(box_crop,pad_width=((0,128-box_w),(0,128-box_h)))
                    )
                just_letter_images=np.array(just_letter_images)
                result = self.letter_detector.predict(just_letter_images) # if this is slow try batching across multiple images. Maybe keep a queue of bbox crop sections and predict on the batch when a threshold is reached?
                print([self.letter_detector.labels[np.argmax(row)] for row in result])

        #     print(tile_index, result)


        # time.sleep(self.SLEEP_TIME)


    def run(self):
        """
        Main run loop for the Imaging pipeline.
        """
        while True:
            self.loop()

if __name__ == "__main__":
    imagingPipeline = Pipeline()
    imagingPipeline.loop()
