# from fieldcapturer import FieldCapturer
# from geolocator import Geolocator
# from targetaggregator import TargetAggregator
# from shapeInference.shape_inference import ShapeInference
# from utils.target import Target

from dataclasses import dataclass
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
import itertools 
import time

# needed if you want to turn on the visualization by commenting out the plot_fns line near the bottom of the loop function
PLOT_RESULT=True
if PLOT_RESULT:
    import shape_detection.src.plot_functions as plot_fns

@dataclass
class ShapeResult:
    shape_label: int
    confidence: float
    bbox: np.ndarray # [min_y, min_x, max_y, max_x] relative to global image coordinates
    tile_index: int

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
        self.shape_model = YOLO("yolo/trained_models/v8n.pt")

        # warm up model
        rand_input = np.random.rand(1, self.tile_resolution, self.tile_resolution,3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)
        # this looks stupid but is necessary because yolov8 only sets up the model on the first call to predict. See site-packages/ultralytics/yolo/engine/model.py in predict() function, inside the `if not self.predictor` block. I profiled it and the setup_model step takes 80% of the time.

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

    def nms_indices(self, boxes: "list[list[int]]", confidences: "list[float]", iou_thresh=0.01):
        '''
        Returns indices of the ones that are duplicates.
        '''
        correct_bboxes = []
        duplicate_indices =  set()
        for i in sorted(range(len(boxes)), key=lambda i: confidences[i], reverse=True):
            x1, y1, x2, y2 = boxes[i]
            is_duplicate = False
            for x3,y3,x4,y4 in correct_bboxes:
                if not (x3>x2 or x1>x4 or y3>y2 or y1>y4):# if they overlap
                    # gets the coordinates of the intersection
                    intersection_x1, intersection_x2 = sorted([x1,x2,x3,x4])[1:3]
                    intersection_y1, intersection_y2 = sorted([y1,y2,y3,y4])[1:3]
                    intersection_area = (intersection_x2-intersection_x1)*(intersection_y2-intersection_y1)
                    a1 = (x2-x1)*(y2-y1)
                    a2 = (x4-x3)*(y4-y3)
                    iou = intersection_area/(a1+a2-intersection_area)
                    if iou >= iou_thresh:
                        is_duplicate=True
            if not is_duplicate:
                correct_bboxes.append(np.array([x1,y1,x2,y2]))
            else:
                duplicate_indices.add(i)
        return duplicate_indices

    def _split_to_tiles(self, img: cv.Mat):
        h,w = img.shape[:2]
        n_horizontal_tiles = np.ceil(w/self.tile_resolution).astype(int)
        n_vertical_tiles = np.ceil(h/self.tile_resolution).astype(int)
        all_tiles = []
        tile_offsets_x_y: 'list[tuple]'  = []
        v_indices = np.linspace(0,h-self.tile_resolution,n_vertical_tiles).astype(int)
        h_indices = np.linspace(0,w-self.tile_resolution,n_horizontal_tiles).astype(int)

        for v,h in itertools.product(v_indices, h_indices):
            tile = img[v:v+self.tile_resolution, h:h+self.tile_resolution]
            all_tiles.append(tile)
            tile_offsets_x_y.append((h,v))

        return (all_tiles, tile_offsets_x_y)

    def _get_letter_crop(self, img: cv.Mat, bbox: 'list[int]'):
        box_x0, box_y0, box_x1, box_y1 = bbox
        box_x1 = min(box_x1, box_x0+self.tile_resolution)
        box_y1 = min(box_y1, box_y0+self.tile_resolution)

        box_crop=img[
            (box_y0):(box_y1),
            (box_x0):(box_x1)
            ]
        
        return np.pad(box_crop,pad_width=((0,128-box_crop.shape[0]),(0,128-box_crop.shape[1])))

    def loop(self):
        # if you need to profile use this: https://stackoverflow.com/a/62382967/14587004

        # ret, img = self.cam.read()
        # if not ret: raise Exception("Failed to grab frame")
        # current_location = self.getCurrentLocation()
        # self.logGeolocation(save_counter, img, current_location)
        img = cv.imread("gopro-image-5k.png")
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        all_tiles, tile_offsets_x_y =self._split_to_tiles(img)

        batch_size = len(all_tiles)

        bboxes_per_tile: "list[Tensor]" = []
        shape_labels, confidences = [], [] 

        # `map(list,...` in this loop makes sure the correct `predict` type overload is being called.
        for batch in np.split(
            ary=all_tiles, 
            indices_or_sections=range(batch_size, len(all_tiles),batch_size),
            axis=0):
            predictions: list[Results] = self.shape_model.predict(list(batch), verbose=False) # TODO: figure out why predict needs batch as a list. I suspect this is a bug and it should be able to take a numpy array, which would be faster
            prediction_tensors: list[Tensor] = [x.to('cpu').boxes.boxes for x in predictions]
            bboxes_per_tile.extend([pred[:,:4] for pred in prediction_tensors])
            shape_labels.extend([[int(x) for x in pred[:,5]+1] for pred in prediction_tensors])
            confidences.extend([pred[:,4] for pred in prediction_tensors])

        all_shape_results: list[ShapeResult] = []

        for tile_index in range(len(all_tiles)):
            for i in range(len(bboxes_per_tile[tile_index])):
                box = bboxes_per_tile[tile_index][i].int().tolist()
                if len(box) == 0:
                    continue
                box[0] += tile_offsets_x_y[tile_index][0]
                box[1] += tile_offsets_x_y[tile_index][1]
                box[2] += tile_offsets_x_y[tile_index][0]
                box[3] += tile_offsets_x_y[tile_index][1]
                all_shape_results.append(
                    ShapeResult(
                        shape_label=shape_labels[tile_index][i],
                        confidence=confidences[tile_index][i],
                        bbox=box,
                        tile_index=tile_index
                    )
                )

        duplicate_indices = self.nms_indices(
            [x.bbox for x in all_shape_results], 
            [x.confidence for x in all_shape_results]
        )

        valid_results: "list[ShapeResult]" = []
        letter_image_buffer=[]
        for i, shape_result in enumerate(all_shape_results):
            if i in duplicate_indices: continue

            letter_image_buffer.append(self._get_letter_crop(grayscale_img, shape_result.bbox))
            valid_results.append(shape_result)
        
        letter_results = self.letter_detector.predict(np.array(letter_image_buffer))
        letter_labels = [self.letter_detector.labels[np.argmax(row)] for row in letter_results]
        if PLOT_RESULT:
            plot_fns.show_image_cv(
                img, 
                [res.bbox for res in valid_results],
                [f"{l}, {self.labels_to_names_dict[x]}" for l,x in zip(letter_labels,[res.shape_label for res in valid_results])],
                [res.confidence for res in valid_results],
                file_name="processed_img.png",
                font_scale=1,thickness=2,box_color=(0,0,255),text_color=(0,0,0))

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