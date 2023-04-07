import time
import json
import itertools
import os
from dataclasses import dataclass


import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.utils import normalize
from ultralytics.yolo.engine.results import Results, Boxes
from ultralytics import YOLO
from std_msgs.msg import Float32MultiArray


from .local_geolocation import GeoLocation
from .color_knn.color_classify import ColorClassifier
from .letter_detection import LetterDetector as letter_detection
from .camera import GoProCamera
from .colordetect.color_segment import color_segmentation
from .best_match import best_match, MATCH_THRESHOLD, CONF_THRESHOLD
from .targetaggregator import TargetAggregator



IMAGING_PATH = os.path.dirname(os.path.realpath(__file__))

# Flag to turn on the visualization
PLOT_RESULT = True
output_folder_path = os.path.join(os.path.dirname(IMAGING_PATH), "flight_data", f"{time.strftime(r'%m-%d-%H-%M-%S')}")
os.makedirs(output_folder_path, exist_ok=True)
from .shape_detection.src import plot_functions as plot_fns


@dataclass
class ShapeResult:
    shape_label: int
    confidence: float
    local_bbox: np.ndarray
    global_bbox: np.ndarray  # [min_y, min_x, max_y, max_x] relative to global image coordinates
    mask: np.ndarray
    tile: np.ndarray


def logGeolocation(loop_index: int, location, heading):
    """
    Save location corresponding to the saved image index.
    """
    f = open(f"{output_folder_path}/locations.txt", "a+")
    f.write(f"Loop index [{loop_index}] has location: [{location}] and heading [{heading}]\n")
    f.close()


def nms_indices(boxes: "list[list[int]]", confidences: "list[float]", iou_thresh=0.01):
    """
    Returns indices of the ones that are duplicates.
    """
    correct_bboxes = []
    duplicates = {}
    duplicate_indices = set()
    for i in sorted(range(len(boxes)), key=lambda i: confidences[i], reverse=True):
        x1, y1, x2, y2 = boxes[i]
        is_duplicate = False
        for j in correct_bboxes:
            x3, y3, x4, y4 = boxes[j]
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
            correct_bboxes.append(i)
            duplicates[i] = []
        else:
            duplicate_indices.add(i)
            duplicates[j].append(i)
    return duplicate_indices,duplicates


def patch_postprocess(self, pp):
    def pp2(*args):
        self.preds.append(args[0])
        return pp(*args)
    return pp2
        

class Pipeline:
    def __init__(self, localizer, img_size, drop_pub, img_file="gopro", targets_file="targets.csv", dry_run=False):
        ''' dry_run being true will just make the pipeline only record the raw images and coordinates and not run any inference'''
        self.doing_dry_run=dry_run
        self.img_file=img_file
        self.localizer = localizer
        self.drop_pub = drop_pub
        if self.img_file == "gopro":
            self.cam = GoProCamera()
        
        if self.doing_dry_run:
            return

        self.geolocator = GeoLocation(img_size)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:  # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

        self.tile_resolution = 640  # has to match img_size of the model, which is determined by which one we use.
        self.shape_model = YOLO(f"{IMAGING_PATH}/yolo/trained_models/seg-v8n.pt", )
        self.letter_detector = letter_detection.LetterDetector(f"{IMAGING_PATH}/trained_model.h5")
        self.color_seg_model = tf.keras.models.load_model(f"{IMAGING_PATH}/colordetect/unet-rgb.hdf5")
        self.color_classifier = ColorClassifier()

        self.target_aggregator = TargetAggregator(targets_file)

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

    def _crop_img(self, img: cv.Mat, bbox: 'list[int]', pad="resize"):
        """
        Args:
            img: Image capture from camera
            bbox: Bounding box of detected target
            pad: None, "resize", or "pad". If None, returns imaging with same dims as img. Otherwise, returns 128x128 image.

        Returns: Cropped image according to bounding box containing only detected target.

        """
        box_x0, box_y0, box_x1, box_y1 = bbox
        if pad == "background":
            box_crop = img[box_y0:box_y0+128,box_x0:box_x0+128]
            return box_crop
        box_crop = img[box_y0:box_y1,box_x0:box_x1]
        if pad is not None:
            if pad=="resize":
                return cv.resize(box_crop.astype(np.float32), (128,128), interpolation=cv.INTER_AREA).astype(np.uint8)
            else:
                pad_widths = [(0, 128 - box_crop.shape[0]), (0, 128 - box_crop.shape[1])]
                if img.ndim == 3:
                    pad_widths.append((0,0))
                return np.pad(box_crop, pad_width=pad_widths)
        else:
            return box_crop

    def _get_image(self):
        """
        Returns: Source image to start the Imaging pipeline
        """
        if self.img_file == "gopro":
            return self.cam.get_image()
        else:
            return cv.imread(self.img_file)
        
    def _get_shape_detections(self, img: cv.Mat, batch_size=1):
        all_tiles, tile_offsets_x_y = self._split_to_tiles(img)

        all_shape_results: list[ShapeResult] = []
        tile_index = 0
        for batch in np.split(
                ary=all_tiles,
                indices_or_sections=range(batch_size, len(all_tiles), batch_size),
                axis=0):
            predictions: list[Results] = self.shape_model.predict(list(batch), verbose=False, conf=MATCH_THRESHOLD)
            # If you don't wrap `batch` in a list it will raise an error.
            # I actually went in and patched this on my local copy of the library in hopes that passing a raw ndarray
            # would make it faster, but it doesn't result in a speedup.
            # with list wrap: 83.81534 seconds for 100 loops
            # without list wrap: 84.9
            prediction_tensors: list[Boxes] = [x.to('cpu') for x in predictions]
            for batch_result, tile_img in zip(prediction_tensors, batch):
                for i, result in enumerate(batch_result.boxes.boxes):
                    box = result[:4].int()
                    global_box = box.detach().clone()
                    global_box[0] += tile_offsets_x_y[tile_index][0]
                    global_box[1] += tile_offsets_x_y[tile_index][1]
                    global_box[2] += tile_offsets_x_y[tile_index][0]
                    global_box[3] += tile_offsets_x_y[tile_index][1]
                    all_shape_results.append(
                        ShapeResult(
                            shape_label=int(result[5]),
                            confidence=result[4],
                            local_bbox=box,
                            global_bbox=global_box,
                            tile=tile_img,
                            mask=batch_result.masks.masks[i].numpy().astype(np.uint8)
                        )
                    )
                tile_index += 1

        duplicate_indices, duplicates = nms_indices(
            [x.global_bbox for x in all_shape_results],
            [x.confidence for x in all_shape_results]
        )

        valid_results: "list[ShapeResult]" = []
        for i, shape_result in enumerate(all_shape_results):
            if i in duplicate_indices: continue
            valid_results.append(shape_result)
            shape_result.duplicates = [all_shape_results[d] for d in duplicates[i]]

        return list(filter(lambda x: x.confidence > CONF_THRESHOLD, valid_results))

    def _get_seg_masks(self, images: np.ndarray) -> np.ndarray:
        '''
        images is of shape (batch_size, 128, 128, 3) 
        '''
        model_input = normalize(images)
        prediction_raw = self.color_seg_model.predict(model_input)
        prediction = np.argmax(prediction_raw, axis=3)
        return prediction
    
    def _get_colors_rgb(self, img_crops: "list[np.ndarray]", masks: np.ndarray):
        """
        Returns (shape_colors, letter_colors)
        where each of those is a list of (ndarrays of shape (3,))
        """
        shape_colors = []
        letter_colors = []
        for img_crop, mask in zip(img_crops, masks):
            shape_colors.append(np.mean(img_crop[mask == 1], axis=0).astype(np.uint8))
            letter_colors.append(np.mean(img_crop[mask == 2], axis=0).astype(np.uint8))
        return shape_colors, letter_colors
    
    def loop(self, loop_index: int):
        # If you need to profile use this: https://stackoverflow.com/a/62382967/14587004
        try:
            cam_img = self._get_image()
            cv.imwrite(f"{output_folder_path}/raw_full{loop_index}.png", cam_img)
            print(f"got image {loop_index}")
            curr_location = self.localizer.get_current_location()
            curr_heading = self.localizer.get_current_heading()
            logGeolocation(loop_index, curr_location, curr_heading)
            
            if self.doing_dry_run:
                return
        except Exception as e:
            print(loop_index)
            print(e)
            return

        valid_results = self._get_shape_detections(cam_img, batch_size=1)

        coords = [
            self.geolocator.get_location(
                res.global_bbox[0], 
                res.global_bbox[1],
                self.localizer.get_current_location(),
                self.localizer.get_current_heading()
            )     
            for res in valid_results
        ]

        if len(valid_results)<1:
            print("no shape detections on index", loop_index)
            return
        print("Finished shape detections")

        if PLOT_RESULT:
            os.makedirs(f"{output_folder_path}/color_seg{loop_index}", exist_ok=True)
        
        color_results = [
            color_segmentation(
                self._crop_img(cv.copyTo(res.tile,res.mask), res.local_bbox, pad=None),
                f"{output_folder_path}/color_seg{loop_index}/{res.shape_label}.png" if PLOT_RESULT else None
            )
            for res in valid_results
        ]


        shape_color_names = [
            self.color_classifier.predict(r.shape_color, bgr=True) for r in color_results
        ]

        letter_color_names = [
            self.color_classifier.predict(r.letter_color, bgr=True) for r in color_results
        ]

        letter_crops = np.array([self._crop_img(cv.copyTo(res.tile,res.mask), res.local_bbox, pad="resize") for res in valid_results])
        USE_UNET = False
        if USE_UNET:
            masks = self._get_seg_masks(letter_crops).astype(np.uint8)
        else:
            letter_masks = [
                cv.resize(res.mask.astype(np.float32), (128,128), interpolation=cv.INTER_AREA).astype(np.uint8)
                for res in color_results
            ]
            masks = np.array(letter_masks).astype(np.uint8)
        
        only_letter_masks = masks*(masks==2) # only takes the letter masks
        only_letter_masks_rgb = np.stack([only_letter_masks]*3, axis=-1)
        letter_image_buffer = cv.copyTo(letter_crops, only_letter_masks_rgb)


        if PLOT_RESULT:
            shape_seg_folder_path = f"{output_folder_path}/shape_seg{loop_index}"
            os.mkdir(shape_seg_folder_path)
            for i, res in enumerate(valid_results):
                cv.imwrite(f"{shape_seg_folder_path}/tile{i}.png", res.tile)
                cv.imwrite(f"{shape_seg_folder_path}/mask{i}.png", res.mask*255)
                combined = cv.copyTo(res.tile,res.mask)
                cv.imwrite(f"{shape_seg_folder_path}/combined{i}.png", combined)
                cv.imwrite(f"{shape_seg_folder_path}/crop{i}.png", self._crop_img(combined, res.local_bbox, pad=None))
            
            seg_folder_path = f"{output_folder_path}/letter_seg{loop_index}"
            os.mkdir(seg_folder_path)
            for i in range(len(letter_image_buffer)):
                cv.imwrite(f"{seg_folder_path}/crop{i}.png", letter_crops[i])
                cv.imwrite(f"{seg_folder_path}/mask{i}.png", masks[i]*127)
                cv.imwrite(f"{seg_folder_path}/letter{i}.png", letter_image_buffer[i])

        letter_results = self.letter_detector.predict(np.mean(letter_image_buffer, axis=-1))
        letter_labels = [self.letter_detector.labels[np.argmax(row)] for row in letter_results]
        print(self.labels_to_names_dict)
        letter_confidences = [list(zip(self.letter_detector.labels, row)) for row in letter_results]
        shape_confidences = [[(self.labels_to_names_dict[i.shape_label], i.confidence) for i in [res] + res.duplicates] for res in valid_results]


        for i in range(len(valid_results)):
            self.target_aggregator.match_target_color(
                coords[i], 
                color_results[i].letter_color, letter_confidences[i], 
                color_results[i].shape_color, shape_confidences[i])



        # shape_colors, letter_colors = self._get_colors_rgb(letter_crops, masks)
        

        if PLOT_RESULT:
            image_file_name = f"{output_folder_path}/det{loop_index}.png"
            plot_fns.show_image_cv(
                cam_img,
                [res.global_bbox for res in valid_results],
                [
                f"{l} | {self.labels_to_names_dict[res.shape_label]} ({res.confidence:.1%}) | Shape Color: {sc} | Letter Color: {lc} | Coords: {c}" for l, res, sc, lc, c in
                 zip(
                    letter_labels, 
                    valid_results,
                    shape_color_names,
                    letter_color_names,
                    coords)],
                file_name=image_file_name,
                font_scale=1, thickness=2, box_color=(0, 0, 255), text_color=(0, 0, 0),
                color_results=color_results
            )

    def run(self, num_loops=50):
        """
        Main run loop for the Imaging pipeline.
        """
        for index in range(num_loops):
            self.loop(index)

        msg = Float32MultiArray()
        msg.data = np.array(self.target_aggregator.target_gps)
        self.drop_pub.publish(msg)



'''
Run commands:

For this one, remember to add @profile on the functions you want to profile, and make sure you did
pip install line_profiler
first

kernprof -l -v main.py
'''
