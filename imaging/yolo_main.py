# from fieldcapturer import FieldCapturer
# from geolocator import Geolocator
# from targetaggregator import TargetAggregator
# from shapeInference.shape_inference import ShapeInference
# from utils.target import Target

import sys
from torch import Tensor

from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO
# from ..navigation.guided_mission.run_mission import Localizer
import letter_detection.LetterDetector as letter_detection

import cv2 as cv
import os
import numpy as np
import json
import tensorflow as tf
import time
# import itertools # needed if you want to turn on the visualization by commenting out the plot_fns line near the bottom of the loop function

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
        backbone_name="mobilenetv2_120d"
        # model = shape_model.EfficientDetModel(
        #     num_classes=13,
        #     img_size=512,
        #     model_architecture=backbone_name # this is the name of the backbone. For some reason it doesn't work with the corresponding efficientdet name.
        #     )
        # model_file=f"shape_detection/trained_models/mobilenetv2_120d_pytorch_25epoch.pt"
        # model.load_state_dict(torch.load(model_file))
        # model.to("cuda")
        # model.eval()
        self.shape_model = YOLO("yolo/trained_models/v8n.pt")

        self.letter_detector = letter_detection.LetterDetector("trained_model.h5")

        with open("./shape_detection/data-gen/shape_name_labels.json","r") as f:
            raw_dict: dict = json.load(f)
            int_casted_keys = map(int, raw_dict.keys())
            self.labels_to_names_dict = dict(zip(int_casted_keys, raw_dict.values()))

        self.targets = [
            ("White", "I", "Brown", "Circle"),
            ("Orange", "V", "Blue", "Rectangle"),
            ("Yellow", "O", "Orange", "Semicircle"),
            ("White", "H", "Red", "Hexagon")
        ]

    
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

    def _split_to_tiles(self, img: cv.Mat):
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
                if any(dim==0 for dim in tile.shape):
                    continue
                all_tiles.append(tile)
                tile_offsets_x_y.append((j*self.tile_resolution,y_offset))
        return (all_tiles, tile_offsets_x_y)

    def _get_letter_crops(self, image, bboxes: 'list[list[float]]'):
        grayscale_image = cv.cvtColor(
            src=np.array(image),
            code=cv.COLOR_RGB2GRAY
        )
        just_letter_images=[]
        for box_x0, box_y0, box_x1, box_y1 in bboxes:
            box_w=int(box_x1)-int(box_x0)
            box_h=int(box_y1)-int(box_y0)
            if box_w>self.tile_resolution or box_h>self.tile_resolution:
                continue
            box_crop=grayscale_image[
                int(box_y0):int(box_y1),
                int(box_x0):int(box_x1)
                ]
            just_letter_images.append(
                np.pad(box_crop,pad_width=((0,128-box_h),(0,128-box_w)))
            )
            # cv.imwrite(f"{str(image)}.png", just_letter_images[-1])
        return np.array(just_letter_images)
    def loop(self):
        # if you need to profile use this: https://stackoverflow.com/a/62382967/14587004

        # ret, img = self.cam.read()
        # if not ret: raise Exception("Failed to grab frame")
        # current_location = self.getCurrentLocation()
        # self.logGeolocation(save_counter, img, current_location)
        img = cv.imread("gopro-image-5k.png")

        all_tiles, tile_offsets_x_y =self._split_to_tiles(img)

        batch_size = len(all_tiles)

        bboxes, shape_labels, confidences = [], [], [] 

        offset_corrected_bboxes = []
        letter_labels = []
        # `map(list,...` in this loop makes sure the correct `predict` type overload is being called.
        for batch in map(list,np.split(
            ary=all_tiles, 
            indices_or_sections=range(batch_size, len(all_tiles),batch_size),
            axis=0)
        ):
            predictions: list[Results] = self.shape_model.predict(batch, verbose=False)
            prediction_tensors: list[Tensor] = [x.to('cpu').boxes.boxes for x in predictions]
            bboxes.extend([pred[:,:4] for pred in prediction_tensors])
            shape_labels.extend([[int(x) for x in pred[:,5]+1] for pred in prediction_tensors])
            confidences.extend([pred[:,4] for pred in prediction_tensors])
        
        letter_image_buffer=None
        for tile_index in range(len(bboxes)):
            if len(bboxes[tile_index])<=0:
                continue

            y_offset,x_offset = tile_offsets_x_y[tile_index]
            just_letter_images = self._get_letter_crops(all_tiles[tile_index], bboxes[tile_index])
            if letter_image_buffer is None:
                letter_image_buffer = just_letter_images
            else:
                letter_image_buffer=np.concatenate([letter_image_buffer,just_letter_images],axis=0)

            for box_x0, box_y0, box_x1, box_y1 in bboxes[tile_index]:
                offset_corrected_bboxes.append([box_x0+x_offset,box_y0+y_offset, box_x1+x_offset, box_y1+y_offset])
        letter_results = self.letter_detector.predict(letter_image_buffer)
        letter_labels = [self.letter_detector.labels[np.argmax(row)] for row in letter_results]
        # plot_fns.show_image_cv(
        #     img, 
        #     offset_corrected_bboxes,
        #     [f"{l}, {self.labels_to_names_dict[x]}" for l,x in zip(letter_labels,itertools.chain(*shape_labels))],
        #     list(itertools.chain(*confidences)),
        #     file_name="processed_img.png",
        #     font_scale=1,thickness=2,box_color=(0,0,255),text_color=(0,0,0))

        #     print(tile_index, result)


        # time.sleep(self.SLEEP_TIME)


    def run(self):
        """
        Main run loop for the Imaging pipeline.
        """
        while True:
            self.loop()

def main():
    imagingPipeline = Pipeline()
    start=time.perf_counter()
    imagingPipeline.loop()
    end=time.perf_counter()
    print(f"elapsed loop time: {end-start:.5f}")

if __name__ == "__main__":
    main()

'''
Run commands:

For this one, remember to add @profile on the functions you want to profile, and make sure you did
pip install line_profiler
first

kernprof -l -v main.py

'''