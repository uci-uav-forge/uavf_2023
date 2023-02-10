from navigation.guided_mission.run_mission import Localizer
from ..letter_detection import LetterDetector
from ..shape_detection.src import shape_det_model

import cv2 as cv
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import time


def logGeolocation(counter: int, img, loc):
    """
    Save image and corresponding location in the savedGeoloc directory.
    The image number corresponds to the save counter in savedGeoloc/locations.txt
    """
    # save image
    img_name = "img{}.png".format(counter)
    cv.imwrite(os.path.join('../savedGeoloc/images', img_name), img)

    # save location
    f = open("../savedGeoloc/locations.txt", "a")
    f.write("Save counter: {} | location: {}\n".format(counter, loc))
    f.close()


class ImagingPipeline:
    VID_CAP_PORT = 1  # Capture port for camera
    SLEEP_TIME = 10  # Picture taken every 10 seconds
    IMAGE_SIZE = 512

    def __init__(self):
        self.tile_resolution = self.IMAGE_SIZE  # has to match img_size of the model, which is determined by which one we use.

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:  # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )

        self.shape_model = shape_det_model.getShapeModel(self.IMAGE_SIZE)
        self.letter_detector = LetterDetector.LetterDetector("trained_model.h5")
        self.labels_to_names_dict = shape_det_model.getLabelToNameDict()
        self.localizer = Localizer()

    def getImageTiles(self, img: cv.Mat):
        """
        Split the image captured by the camera into tiles for shape detection processing.
        Args:
            img: Image captured by camera

        Returns: List of tiles and their (x,y) offsets
        """
        h, w = img.shape[:2]
        n_horizontal_tiles = w // self.tile_resolution
        n_vertical_tiles = h // self.tile_resolution
        all_tiles = []
        h_tiles = np.split(img, range(self.tile_resolution, (n_horizontal_tiles + 1) * self.tile_resolution,
                                      self.tile_resolution), axis=1)
        tile_offsets_x_y: 'list[tuple]' = []

        for i, h_tile in enumerate(h_tiles):
            y_offset = i * self.tile_resolution
            v_tiles = np.split(h_tile, range(self.tile_resolution, (n_vertical_tiles + 1) * self.tile_resolution,
                                             self.tile_resolution), axis=0)
            for j, tile in enumerate(v_tiles):
                if any(dim == 0 for dim in tile.shape):
                    continue
                all_tiles.append(tile)
                tile_offsets_x_y.append((j * self.tile_resolution, y_offset))
        return all_tiles, tile_offsets_x_y

    def getLetterCrops(self, image: Image, bboxes: 'list[list[float]]'):
        """
        Args:
            image: Captured by the camera.
            bboxes: Bounding box of each detected target. Image is cropped according to these bounding boxes

        Returns: Cropped image containing just the letter for the letter recognition model
        """
        grayscale_image = cv.cvtColor(
            src=np.array(image),
            code=cv.COLOR_RGB2GRAY
        )
        just_letter_images = []
        for box_x0, box_y0, box_x1, box_y1 in bboxes:
            box_w = int(box_x1) - int(box_x0)
            box_h = int(box_y1) - int(box_y0)
            if box_w > self.tile_resolution or box_h > self.tile_resolution:
                continue
            box_crop = grayscale_image[
                       int(box_y0):int(box_y1),
                       int(box_x0):int(box_x1)
                       ]
            just_letter_images.append(
                np.pad(box_crop, pad_width=((0, 128 - box_h), (0, 128 - box_w)))
            )
        return np.array(just_letter_images)

    def getCurrentLocation(self):
        """
        Return the current local location of the UAV.
        """
        return self.localizer.get_current_location()

    def captureImage(self):
        """
        Returns: Capture the image with camera, treating the camera as a webcam.
        """
        # ret, img = self.cam.read()
        # if not ret: raise Exception("Failed to grab frame")
        # current_location = self.getCurrentLocation()
        # self.logGeolocation(save_counter, img, current_location)
        return cv.imread("gopro-image-5k.png")

    def loop(self):
        # if you need to profile use this: https://stackoverflow.com/a/62382967/14587004

        img = self.captureImage()
        all_tiles, tile_offsets_x_y = self.getImageTiles(img)
        pil_images = [Image.fromarray(tile) for tile in all_tiles]
        batch_size = 8
        bboxes, labels, confidences = [], [], []
        offset_corrected_bboxes = []
        # `map(list,...` in this loop makes sure the correct `predict` type overload is being called.
        for batch in map(list, np.split(
                ary=pil_images,
                indices_or_sections=range(batch_size, len(pil_images), batch_size),
                axis=0)
                         ):
            res = self.shape_model.predict(batch)
            b, l, c = res
            bboxes.extend(b)
            labels.extend(l)
            confidences.extend(c)

        letter_image_buffer = None
        for tile_index in range(len(bboxes)):
            if len(bboxes[tile_index]) <= 0:
                continue

            y_offset, x_offset = tile_offsets_x_y[tile_index]

            letter_only_images = self.getLetterCrops(pil_images[tile_index], bboxes[tile_index])
            if letter_image_buffer is None:
                letter_image_buffer = letter_only_images
            else:
                letter_image_buffer = np.concatenate([letter_image_buffer, letter_only_images], axis=0)

            for box_x0, box_y0, box_x1, box_y1 in bboxes[tile_index]:
                offset_corrected_bboxes.append(
                    [box_x0 + x_offset, box_y0 + y_offset, box_x1 + x_offset, box_y1 + y_offset])

        letter_results = self.letter_detector.predict(letter_image_buffer)
        letter_labels = [self.letter_detector.labels[np.argmax(row)] for row in letter_results]

        # plot_fns.show_image_cv(
        #     img, 
        #     offset_corrected_bboxes,
        #     [f"{l}, {self.labels_to_names_dict[x]}" for l,x in zip(letter_labels,itertools.chain(*labels))],
        #     list(itertools.chain(*confidences)),
        #     file_name="processed_img.png",
        #     font_scale=1,thickness=2,box_color=(0,0,255),text_color=(0,0,0))

        #     print(tile_index, result)

    def run(self):
        """
        Main run loop for the Imaging pipeline.
        """
        while True:
            self.loop()
            time.sleep(self.SLEEP_TIME)  # must be greater than the time it takes for loop() to complete


def main():
    imagingPipeline = ImagingPipeline()
    start = time.perf_counter()
    imagingPipeline.loop()
    end = time.perf_counter()
    print(f"elapsed loop time: {end - start:.5f}")


if __name__ == "__main__":
    main()

'''
Run commands:
For this one, remember to add @profile on the functions you want to profile, and make sure you did
pip install line_profiler
first
kernprof -l -v main.py

'''
