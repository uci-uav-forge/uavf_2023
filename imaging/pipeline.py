import time
import json
import itertools
import os
from dataclasses import dataclass
import rospy
import traceback as tb
import queue

import cv2 as cv
import numpy as np
from ultralytics.yolo.engine.results import Results, Boxes
from ultralytics import YOLO
from std_msgs.msg import String, Bool


from .local_geolocation import GeoLocator 
from .color_knn.color_classify import ColorClassifier
from .camera import GoProCamera
from .zone_coverage import ZoneCoverageTracker
from .colordetect.color_segment import color_segmentation
from .best_match import best_match, MATCH_THRESHOLD, CONF_THRESHOLD
from .targetaggregator import TargetAggregator
from .shape_detection.src import plot_functions as plot_fns
from navigation.mock_drone import MockDrone
from tqdm import tqdm
import threading

IMAGING_PATH = os.path.dirname(os.path.realpath(__file__))

# Flag to turn on the visualization
PLOT_RESULT = True
output_folder_path = os.path.join(os.path.dirname(IMAGING_PATH), "flight_data", f"{time.strftime(r'%m-%d-%H-%M-%S')}")
os.makedirs(output_folder_path, exist_ok=True)

index=0 # defined globally so we don't get overlaps on file names when the pipeline starts, stops, then restarts itself

@dataclass
class ShapeResult:
    shape_label: int
    confidence: float
    local_bbox: np.ndarray
    global_bbox: np.ndarray  # [min_y, min_x, max_y, max_x] relative to global image coordinates
    mask: np.ndarray
    tile: np.ndarray

unique_labels = set()
def nms_indices(boxes: "list[list[int]]", confidences: "list[float]", iou_thresh=0.01):
    """
    Returns indices of the ones that are duplicates for non-max suppression.
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
    return duplicate_indices, duplicates

class MockCamera:
    def __init__(self, folder_name):
        self.idx = 0
        self.folder_name = folder_name

    def get_image(self):
        img = cv.imread(f"{IMAGING_PATH}/../gopro_tests/{self.folder_name}/img{self.idx}.png")
        if img is None:
            print("ran out of images")
            return None
        self.idx += 1
        return img


def get_colors_rgb(img_crops: "list[np.ndarray]", masks: np.ndarray):
    """
    Returns (shape_colors, letter_colors)
    where each of those is a list of (ndarrays of shape (3,))
    """
    shape_colors = list()
    letter_colors = list()
    for img_crop, mask in zip(img_crops, masks):
        shape_colors.append(np.mean(img_crop[mask == 1], axis=0).astype(np.uint8))
        letter_colors.append(np.mean(img_crop[mask == 2], axis=0).astype(np.uint8))
    return shape_colors, letter_colors


def crop_image(img: cv.Mat, bbox: 'list[int]', pad="resize"):
    """
    Args:
        img: Image capture from camera
        bbox: Bounding box of detected target
        pad: None, "resize", or "pad". If None, returns imaging with same dims as img. Otherwise, returns 128x128 image.

    Returns: Cropped image according to bounding box containing only detected target.
    """
    box_x0, box_y0, box_x1, box_y1 = bbox
    if pad == "background":
        box_crop = img[box_y0:box_y0 + 128, box_x0:box_x0 + 128]
        return box_crop
    box_crop = img[box_y0:box_y1, box_x0:box_x1]
    if pad is not None:
        if pad == "resize":
            return cv.resize(box_crop.astype(np.float32), (128, 128), interpolation=cv.INTER_AREA).astype(np.uint8)
        else:
            pad_widths = [(0, 128 - box_crop.shape[0]), (0, 128 - box_crop.shape[1])]
            if img.ndim == 3:
                pad_widths.append((0, 0))
            return np.pad(box_crop, pad_width=pad_widths)
    else:
        return box_crop


class Pipeline:
    def __init__(self, drone: MockDrone, drop_pub: rospy.Publisher, drop_zone_coords: np.ndarray= None, drop_sub = False, img_file="gopro", targets_file="targets.csv", dry_run=False):
        """ dry_run being true will just make the pipeline only record the raw images and coordinates and
        not run any inference
        """
        self.doing_dry_run = dry_run
        if drop_zone_coords is None:
            print("no drop zone coords specified, using 50x50 box around current position")
            drop_zone_coords = np.array([(-50, -50), (50, -50), (50, 50), (-50, 50)]) + np.array(drone.get_current_xyz()[:2])
        self.zone_coverage_tracker = ZoneCoverageTracker(
            dropzone_local_coords=drop_zone_coords
        ) 
        self.dz_bounds_sub = rospy.Subscriber(name="dropzone_bounds", data_class=String, callback=self.drop_bounds_cb)
        self.img_file = img_file
        self.drone = drone
        self.drop_pub = drop_pub
        if drop_sub:
            self.drop_sub = rospy.Subscriber(name="drop_signal", data_class=Bool, callback=self.drop_sub_cb)
            self.drop = False
        else:
            self.drop_sub = None
        if self.img_file == "gopro":
            self.cam = GoProCamera()
        elif not self.img_file.endswith(".png") and not self.img_file.endswith(".jpg"):
            self.cam = MockCamera(self.img_file)
        else:
            self.cam = None
        if self.doing_dry_run: return

        self.geolocator = GeoLocator()

        """ gpus = tf.config.list_physical_devices('GPU')
        if gpus:  # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]) """

        self.tile_resolution = 640  # has to match img_size of the model, which is determined by which one we use.
        self.shape_model = YOLO(f"{IMAGING_PATH}/yolo/trained_models/seg-v8n.pt", )
        #self.letter_detector = letter_detection.LetterDetector(f"{IMAGING_PATH}/trained_model.h5")
        self.letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","1","2","3","4","5","6","7","8","9"]
        self.letter_detector = YOLO(f"{IMAGING_PATH}/yolo/trained_models/letter.pt")
        #self.color_seg_model = tf.keras.models.load_model(f"{IMAGING_PATH}/colordetect/unet-rgb.hdf5")
        self.color_classifier = ColorClassifier()

        self.target_aggregator = TargetAggregator(targets_file)

        # warm up shape model
        rand_input = np.random.rand(1, self.tile_resolution, self.tile_resolution, 3).astype(np.float32)
        self.shape_model.predict(list(rand_input), verbose=False)
        self.letter_detector.predict(list(rand_input), verbose=False)
        # YOLOv8 only sets up the model on the first call to predict.
        # See site-packages/ultralytics/yolo/engine/model.py in predict() function,
        # inside the `if not self.predictor` block. I profiled it and the setup_model step takes 80% of the time.

        with open(f"{IMAGING_PATH}/shape_detection/data-gen/shape_name_labels.json", "r") as f:
            raw_dict: dict = json.load(f)
            int_casted_keys = map(int, raw_dict.keys())
            self.labels_to_names_dict = dict(zip(int_casted_keys, raw_dict.values()))

        # record each pipeline iteration results
        self.loop_index = None
        self.valid_results = None

    def drop_bounds_cb(self, msg: String):
        print(f"re-initializing dropzone bounds: {msg.data}")
        drop_zone_coords = json.loads(msg.data)
        self.zone_coverage_tracker = ZoneCoverageTracker(
            dropzone_local_coords=np.array(drop_zone_coords)
        )

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

    def _get_image(self):
        """
        Returns: Source image to start the Imaging pipeline
        """
        if self.cam is not None: return self.cam.get_image()
        else: return cv.imread(self.img_file)

    def _get_shape_detections(self, img: cv.Mat, batch_size=1, num_results=10):
        all_tiles, tile_offsets_x_y = self._split_to_tiles(img)

        all_shape_results: list[ShapeResult] = []
        tile_index = 0
        for batch in tqdm(np.split(
                ary=all_tiles,
                indices_or_sections=range(batch_size, len(all_tiles), batch_size),
                axis=0)):
            predictions: list[Results] = self.shape_model.predict(list(batch), verbose=False, conf=MATCH_THRESHOLD)
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

        self.valid_results: "list[ShapeResult]" = list()
        for i, shape_result in enumerate(all_shape_results):
            if i in duplicate_indices: continue
            self.valid_results.append(shape_result)
            shape_result.duplicates = [all_shape_results[d] for d in duplicates[i]]

        return list(filter(lambda x: x.confidence > CONF_THRESHOLD, self.valid_results))

    def _plotSegmentationsAndMask(self, letter_image_buffer, letter_crops, masks):
        shape_seg_folder_path = f"{output_folder_path}/shape_seg{self.loop_index}"
        os.mkdir(shape_seg_folder_path)
        for i, res in enumerate(self.valid_results):
            cv.imwrite(f"{shape_seg_folder_path}/tile{i}.png", res.tile)
            cv.imwrite(f"{shape_seg_folder_path}/mask{i}.png", res.mask * 255)
            combined = cv.copyTo(res.tile, res.mask)
            cv.imwrite(f"{shape_seg_folder_path}/combined{i}.png", combined)
            cv.imwrite(f"{shape_seg_folder_path}/crop{i}.png", crop_image(combined, res.local_bbox, pad=None))

        seg_folder_path = f"{output_folder_path}/letter_seg{self.loop_index}"
        os.mkdir(seg_folder_path)
        for i in range(len(letter_image_buffer)):
            cv.imwrite(f"{seg_folder_path}/crop{i}.png", letter_crops[i])
            cv.imwrite(f"{seg_folder_path}/mask{i}.png", masks[i] * 127)
            cv.imwrite(f"{seg_folder_path}/letter{i}.png", letter_image_buffer[i])

    def _plotPipelineResults(self, cam_img, letter_labels, shape_color_names, coords, letter_color_names, color_results):
        image_file_name = f"{output_folder_path}/det{self.loop_index}.png"
        plot_fns.show_image_cv(
            cam_img,
            [res.global_bbox for res in self.valid_results],
            [
                f"{l} | {self.labels_to_names_dict[res.shape_label]} ({res.confidence:.1%}) | Shape Color: {sc} | Letter Color: {lc} | Coords: {c}"
                for l, res, sc, lc, c in
                zip(
                    letter_labels,
                    self.valid_results,
                    shape_color_names,
                    letter_color_names,
                    coords)],
            file_name=image_file_name,
            font_scale=1, thickness=2, box_color=(0, 0, 255), text_color=(0, 0, 0),
            color_results=color_results
        )
    
    def _logLocation(self, coords, angles):
        with open(f"{output_folder_path}/coords.txt", "a") as f:
            f.write(f"Coords: {coords[0]}, {coords[1]}, {coords[2]}\n")
            f.write(f"Angles: {angles[0]}, {angles[1]}, {angles[2]}\n")

    def loop(self, loop_index: int):
        self.loop_img(loop_index, self._get_image())
    
    def loop_img(self, loop_index: int, img):
        # If you need to profile use this: https://stackoverflow.com/a/62382967/14587004
        self.loop_index = loop_index
        try:
            print(f"Getting image {loop_index}")
            cam_img = img#self._get_image()
            cv.imwrite(f"{output_folder_path}/image{loop_index}.png", cam_img)
            print(f"got image {loop_index}")
            curr_location, curr_angles = self.drone.get_current_pos_and_angles()
            self._logLocation(curr_location, curr_angles)
            self.zone_coverage_tracker.add_coverage(curr_location, curr_angles)
            if self.doing_dry_run: return
        except Exception as e:
            print(f"Exception on pipeline loop {loop_index} with error {e}")
            tb.print_exc()
            return

        self.valid_results = self._get_shape_detections(cam_img, batch_size=1)
        coords = [
            self.geolocator.get_location(
                res.global_bbox[0],
                res.global_bbox[1],
                location=curr_location,
                angles=curr_angles,
                img_size=cam_img.shape[:2]
            )
            for res in self.valid_results
        ]

        if len(self.valid_results) < 1:
            print("no shape detections on index", loop_index)
            return
        print("Finished shape detections")

        if PLOT_RESULT: os.makedirs(f"{output_folder_path}/color_seg{loop_index}", exist_ok=True)

        color_results = [
            color_segmentation(
                crop_image(cv.copyTo(res.tile, res.mask), res.local_bbox, pad=None),
                f"{output_folder_path}/color_seg{loop_index}/{res.shape_label}.png" if PLOT_RESULT else None
            )
            for res in self.valid_results
        ]

        shape_color_names = [self.color_classifier.predict(r.shape_color, bgr=True) for r in color_results]
        letter_color_names = [self.color_classifier.predict(r.letter_color, bgr=True) for r in color_results]
        letter_crops = np.array(
            [crop_image(cv.copyTo(res.tile, res.mask), res.local_bbox, pad="resize") for res in self.valid_results]
        )

        USE_UNET = False
        if USE_UNET:
            masks = self._get_seg_masks(letter_crops).astype(np.uint8)
        else:
            letter_masks = [
                cv.resize(res.mask.astype(np.float32), (128, 128), interpolation=cv.INTER_AREA).astype(np.uint8)
                for res in color_results
            ]
            masks = np.array(letter_masks).astype(np.uint8)

        only_letter_masks = masks * (masks == 2)  # only takes the letter masks
        only_letter_masks_rgb = np.stack([only_letter_masks] * 3, axis=-1)
        letter_image_buffer = cv.copyTo(letter_crops, only_letter_masks_rgb)

        if PLOT_RESULT: self._plotSegmentationsAndMask(letter_image_buffer, letter_crops, masks)

        letter_results = []
        letter_labels = []
        for i in range(len(self.valid_results)):
            result = self.letter_detector.predict(letter_image_buffer[i], verbose=False, conf=MATCH_THRESHOLD)
            letter_results.append(result)
            classification = np.argmax(result[0].probs.numpy())
            letter_labels.append(self.letters[int(self.letter_detector.names[classification])])
        letter_confidences = [list(zip(self.letters, row[0].probs)) for row in letter_results]
        shape_confidences = [[(self.labels_to_names_dict[i.shape_label], i.confidence) for i in [res] + res.duplicates]
                             for res in self.valid_results]
        
        for i in range(len(self.valid_results)):
            self.target_aggregator.match_target_color(
                coords[i],
                color_results[i].letter_color, letter_confidences[i],
                color_results[i].shape_color, shape_confidences[i])

        # shape_colors, letter_colors = self._get_colors_rgb(letter_crops, masks)

        if PLOT_RESULT:
            self._plotPipelineResults(cam_img, letter_labels, shape_color_names, coords, letter_color_names, color_results)
            cv.imwrite(f"{output_folder_path}/coverage{loop_index}.png", self.zone_coverage_tracker.get_coverage_image())

    def drop_sub_cb(self, data):
        print(f"Drop sub received with data: {data}")
        self.drop = data.data

    def run(self, num_loops=1):
        global index
        """
        Main run loop for the Imaging pipeline.
        """
        if self.drop_sub is None:
            for _i in range(num_loops):
                self.loop(index)
                index+=1
        else:
            while 1:
                print("Listening for drop signal")
                while not self.drop:
                    time.sleep(0.1)
                print("Drop signal received")
                while self.drop:
                    self.loop(index)
                    index += 1
                msg = String()
                valid_target_coords_with_indices = []
                for i, coord in enumerate(self.target_aggregator.get_target_coords()):
                    if coord is None: 
                        print(f"Could not find target {i}")
                        continue
                    valid_target_coords_with_indices.append((coord[0], coord[1], i))
                msg.data = json.dumps(valid_target_coords_with_indices)
                self.drop_pub.publish(msg)
                print(f"Published drop message: {msg.data}")
    
    def run_concurrent(self):
        REQUIRED_PCT_UNCOVERED = 1
        IMG_H_W_METERS = (3,4)


        done = False
        img_queue = queue.PriorityQueue()

        def loop():
            while not done:
                coverage, idx, next_img = img_queue.get(timeout=2) if not img_queue.empty() else None
                if next_img is None:
                    continue
                self.loop_img(idx, next_img)
                idx += 1
        
        self.loop_thread = threading.Thread(target = loop)
        while not self.drop:
            time.sleep(0.1)
        self.loop_thread.start()
        idx = 0
        while self.drop:
            curr_location, curr_angles = self.drone.get_current_pos_and_angles()
            coverage = self.zone_coverage_tracker.get_point_coverage(curr_location, curr_angles, IMG_H_W_METERS) 
            if coverage < REQUIRED_PCT_UNCOVERED:
                i_nx = self._get_image()
                self.zone_coverage_tracker.add_coverage(curr_location, curr_angles, IMG_H_W_METERS)
                idx += 1
                img_queue.put((coverage,idx,i_nx))
            time.sleep(1)
            

            
        done = True
        loop_thread.join()

        msg = String()
        valid_target_coords_with_indices = []
        for i, coord in enumerate(self.target_aggregator.get_target_coords()):
            if coord is None: 
                print(f"Could not find target {i}")
                continue
            valid_target_coords_with_indices.append((coord[0], coord[1], i))
        msg.data = json.dumps(valid_target_coords_with_indices)
        self.drop_pub.publish(msg)
        print(f"Published drop message: {msg.data}")
    



           




'''
Run commands:

For this one, remember to add @profile on the functions you want to profile, and make sure you did
pip install line_profiler
first

kernprof -l -v main.py
'''
