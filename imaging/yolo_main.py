from dataclasses import dataclass
from torch import Tensor

from ultralytics.yolo.engine.results import Results
from ultralytics import YOLO
from .letter_detection import LetterDetector as letter_detection
from .camera import GoProCamera

import cv2 as cv
import numpy as np
import json
import tensorflow as tf
import itertools
import time
import os

# Flag to turn on the visualization
# PLOT_RESULT = False
PLOT_RESULT = True
if PLOT_RESULT:
    from .shape_detection.src import plot_functions as plot_fns

IMAGING_PATH = os.path.dirname(os.path.realpath(__file__))

@dataclass
class ShapeResult:
    shape_label: int
    confidence: float
    bbox: np.ndarray  # [min_y, min_x, max_y, max_x] relative to global image coordinates
    tile_index: int


def logGeolocation(counter: int, location):
    """
    Save location corresponding to the saved image index.
    """
    f = open("locations.txt", "w")
    f.write("Save counter [{}] with location: [{}]\n".format(counter, location))
    f.close()


def nms_indices(boxes: "list[list[int]]", confidences: "list[float]", iou_thresh=0.01):
    """
    Returns indices of the ones that are duplicates.
    """
    correct_bboxes = []
    duplicate_indices = set()
    for i in sorted(range(len(boxes)), key=lambda i: confidences[i], reverse=True):
        x1, y1, x2, y2 = boxes[i]
        is_duplicate = False
        for x3, y3, x4, y4 in correct_bboxes:
            if not (x3 > x2 or x1 > x4 or y3 > y2 or y1 > y4):  # if they overlap
                # gets the coordinates of the intersection
                intersection_x1, intersection_x2 = sorted([x1, x2, x3, x4])[1:3]
                intersection_y1, intersection_y2 = sorted([y1, y2, y3, y4])[1:3]
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (x4 - x3) * (y4 - y3)
                iou = intersection_area / (a1 + a2 - intersection_area)
                if iou >= iou_thresh:
                    is_duplicate = True
        if not is_duplicate:
            correct_bboxes.append(np.array([x1, y1, x2, y2]))
        else:
            duplicate_indices.add(i)
    return duplicate_indices


class Pipeline:
    def __init__(self, localizer, cam_mode="gopro"):
        self.cam_mode=cam_mode

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:  # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

        self.tile_resolution = 640  # has to match img_size of the model, which is determined by which one we use.
        self.shape_model = YOLO(f"{IMAGING_PATH}/yolo/trained_models/v8n-640.pt")
        self.letter_detector = letter_detection.LetterDetector(f"{IMAGING_PATH}/trained_model.h5")
        self.localizer = localizer
        if self.cam_mode == "gopro":
            self.cam = GoProCamera()

        # warm up shape model
        rand_input = np.random.rand(1, self.tile_resolution, self.tile_resolution, 3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)
        # this looks stupid but is necessary because yolov8 only sets up the model on the first call to predict.
        # See site-packages/ultralytics/yolo/engine/model.py in predict() function,
        # inside the `if not self.predictor` block. I profiled it and the setup_model step takes 80% of the time.

        with open(f"{IMAGING_PATH}/shape_detection/data-gen/shape_name_labels.json", "r") as f:
            raw_dict: dict = json.load(f)
            int_casted_keys = map(int, raw_dict.keys())
            self.labels_to_names_dict = dict(zip(int_casted_keys, raw_dict.values()))

    def getCurrentLocation(self):
        """
        Return the current local location of the UAV.
        """
        return self.localizer.get_current_location()

    def _split_to_tiles(self, img: cv.Mat):
        h, w = img.shape[:2]
        n_horizontal_tiles = np.ceil(w / self.tile_resolution).astype(int)
        n_vertical_tiles = np.ceil(h / self.tile_resolution).astype(int)
        all_tiles: list[np.ndarray] = []
        tile_offsets_x_y: 'list[tuple]' = []
        v_indices = np.linspace(0, h - self.tile_resolution, n_vertical_tiles).astype(int)
        h_indices = np.linspace(0, w - self.tile_resolution, n_horizontal_tiles).astype(int)

        for v, h in itertools.product(v_indices, h_indices):
            tile = img[v:v + self.tile_resolution, h:h + self.tile_resolution]
            all_tiles.append(tile)
            tile_offsets_x_y.append((h, v))

        return all_tiles, tile_offsets_x_y

    def _get_letter_crop(self, img: cv.Mat, bbox: 'list[int]'):
        """
        Args:
            img: Reformatted image capture from camera
            bbox: Bounding box of detected target

        Returns: Cropped image according to bounding box containing only detected target.

        """
        box_x0, box_y0, box_x1, box_y1 = bbox
        box_x1 = min(box_x1, box_x0 + self.tile_resolution)
        box_y1 = min(box_y1, box_y0 + self.tile_resolution)

        box_crop = img[
                   (box_y0):(box_y1),
                   (box_x0):(box_x1)
                   ]

        return np.pad(box_crop, pad_width=((0, 128 - box_crop.shape[0]), (0, 128 - box_crop.shape[1])))

    def _get_image(self):
        """
        Returns: Source image to start the Imaging pipeline
        """
        if self.cam_mode == "gopro":
            return self.cam.get_image()
        elif self.cam_mode == "image":
            return cv.imread(f"{IMAGING_PATH}/gopro-image-5k.png")

    def loop(self, index: int):
        # If you need to profile use this: https://stackoverflow.com/a/62382967/14587004
        img = self._get_image()
        cv.imwrite(f"raw_img{index}.png", img)
        print(f"got image {index}")
        curr_location = self.getCurrentLocation()
        logGeolocation(index, curr_location)

        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        all_tiles, tile_offsets_x_y = self._split_to_tiles(img)

        # batch_size = len(all_tiles)
        batch_size = 1  # when running on Jetson Nano

        bboxes_per_tile: "list[Tensor]" = []
        shape_labels, confidences = [], []
        all_shape_results: list[ShapeResult] = []
        tile_index = 0
        for batch in np.split(
                ary=all_tiles,
                indices_or_sections=range(batch_size, len(all_tiles), batch_size),
                axis=0):
            predictions: list[Results] = self.shape_model.predict(list(batch), verbose=False)
            # If you don't wrap `batch` in a list it will raise an error.
            # I actually went in and patched this on my local copy of the library in hopes that passing a raw ndarray
            # would make it faster, but it doesn't result in a speedup.
            # with list wrap: 83.81534 seconds for 100 loops
            # without list wrap: 84.9
            prediction_tensors: list[Tensor] = [x.to('cpu').boxes.boxes for x in predictions]
            for batch_result in prediction_tensors:
                for result in batch_result:
                    box = result[:4].int().tolist()
                    box[0] += tile_offsets_x_y[tile_index][0]
                    box[1] += tile_offsets_x_y[tile_index][1]
                    box[2] += tile_offsets_x_y[tile_index][0]
                    box[3] += tile_offsets_x_y[tile_index][1]
                    all_shape_results.append(
                        ShapeResult(
                            shape_label=int(result[5]),
                            confidence=result[4],
                            bbox=box,
                            tile_index=tile_index
                        )
                    )
                tile_index += 1

        duplicate_indices = nms_indices(
            [x.bbox for x in all_shape_results],
            [x.confidence for x in all_shape_results]
        )

        valid_results: "list[ShapeResult]" = []
        letter_image_buffer = []
        for i, shape_result in enumerate(all_shape_results):
            if i in duplicate_indices: continue

            letter_image_buffer.append(self._get_letter_crop(grayscale_img, shape_result.bbox))
            valid_results.append(shape_result)

        if len(letter_image_buffer)<1:
            print("no shape detections on index", index)
            return

        letter_results = self.letter_detector.predict(np.array(letter_image_buffer))
        letter_labels = [self.letter_detector.labels[np.argmax(row)] for row in letter_results]
        if PLOT_RESULT:
            image_file_name = "detection_results_num{}.jpg".format(index)
            plot_fns.show_image_cv(
                img,
                [res.bbox for res in valid_results],
                [f"{l}, {self.labels_to_names_dict[x]}" for l, x in
                 zip(letter_labels, [res.shape_label for res in valid_results])],
                [res.confidence for res in valid_results],
                file_name=image_file_name,
                font_scale=1, thickness=2, box_color=(0, 0, 255), text_color=(0, 0, 0)
            )

    def run(self, num_loops=50):
        """
        Main run loop for the Imaging pipeline.
        """
        for index in range(num_loops):
            self.loop(index)

'''
Run commands:

For this one, remember to add @profile on the functions you want to profile, and make sure you did
pip install line_profiler
first

kernprof -l -v main.py
'''
