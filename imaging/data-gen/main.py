import math
import os
import json
import random
from typing import Callable

import cv2
import numpy as np
from  image_rotation import get_rotated_image, get_shape_bbox

def create_shape_dataset(get_frame: Callable[[], cv2.Mat], 
                         shapes_directory:str, 
                         shape_resolution: int = 36, 
                         max_shapes_per_image: int = 3, 
                         num_images:int = 100,
                         blur_radius=3,
                         data_split=[0.7,0.2,0.1]):
    '''
    creates a directory called "output" adjacent to wherever this is run from and fills it with output images.

    ARGUMENTS:

    `get_frame` is a function that returns an image (cv2 mat). This could just return the same image every time, a random image from a directory, or a frame from a video. The implementation doesn't matter as long as the return type is consistent.

    `shapes_directory` contains the images of the shapes to apply to the background.

    `shape_resolution` is how big each shape will be drawn on the background, in pixels

    `max_shapes_per_image` is how many shapes are on each image, max. It will choose a number between 0 and this parameter for each output image.
    
    `num_images` is how many images to output

    `blur_radius` is the size of the gaussian kernel. The default is 3, which means a 3x3 kernel.

    `data_split` is a 3-tuple describing the proportions of train, validation, and test data generated respectively.
    '''
    shapes = dict(
        (
            name.split(".")[0], 
            cv2.cvtColor(cv2.imread(f'{shapes_directory}/{name}'),cv2.COLOR_BGR2GRAY)
        ) 
        for name in os.listdir(shapes_directory)
    )
    categories = [
        {
            "id": i,
            "name": name,
            "supercategory": "none"
        } for i,name in enumerate(shapes.keys())
    ]
    if "output" not in os.listdir():
        os.mkdir("output")
        os.mkdir("output/train")
        os.mkdir("output/validation")
        os.mkdir("output/test")
    image_idx=0
    for num_in_split, split_dir_name in [(data_split[0]*num_images, "train"),(data_split[1]*num_images, "validation"), (data_split[2]*num_images, "test")]:
        annotations = []
        images = []
        for _ in range(math.ceil(num_in_split)):
            image_idx+=1
            frame=get_frame()
            height, width = frame.shape[:2]
            # shape: (height, width, 3) e.g. (2988, 5312, 3)
            for shape_idx in range(random.randint(0,max_shapes_per_image)):
                shape_name, category_num = random.choice(list(zip(shapes.keys(), range(len(shapes)))))
                shape_source_h, shape_source_w = shapes[shape_name].shape[:2]
                shape_resize_w = shape_resolution # the shape source image's width and height
                shape_to_draw = get_rotated_image(
                    img=cv2.resize(shapes[shape_name], (shape_resize_w, int(shape_resize_w*shape_source_h/shape_source_w))),
                    theta=np.random.random()*np.pi*2
                )
                bbox = get_shape_bbox(shape_to_draw)

                shape_h, shape_w = bbox[3]-bbox[1], bbox[2]-bbox[0]
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                x_offset = random.randint(0,width-shape_w)
                y_offset = random.randint(0, height-shape_h)
                for y in range(1,shape_h):
                    for x in range(1,shape_w):
                        if shape_to_draw[bbox[0]+x][bbox[1]+y] > 0 :
                            frame[y+y_offset][x+x_offset] = color
                frame=cv2.blur(frame, (blur_radius, blur_radius))
                annotations.append({
                    "id": image_idx*max_shapes_per_image+shape_idx,
                    "image_id": image_idx,
                    "category_id": category_num,
                    "bbox": [(x_offset),(y_offset),shape_w,shape_h],
                    "area": shape_w*shape_h,
                    "segmentation": [],
                    "iscrowd": 0
                })
            output_file_name = f"image{image_idx}.png"
            cv2.imwrite(f"./output/{split_dir_name}/{output_file_name}", frame)
            print(f"Finished {image_idx}/{num_images}")
            images.append({
                "id": image_idx,
                "license": 1,
                "file_name": output_file_name,
                "height": height,
                "width": width,
                "date_captured":"2022-11-10T12:00:00+00:00"
            })
            # cv2.imshow('frame', cv2.resize(frame, (1600, 900)))
            # if cv2.waitKey(0) == ord('q'):
            #     break

        # after generating images, write metadata to coco.json
        coco_metadata = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {
                "year": "2022",
                "version": 1,
                "description": f"UAV Forge shapes dataset with {shape_resolution} pixels per shape, {max_shapes_per_image} shapes per image max, and {num_images} images total.",
                "contributor": "Eric Pedley"
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://opensource.org/licenses/MIT",
                    "name": "MIT"
                }
            ]
        }
        with open(f"output/{split_dir_name}/coco.json", "x") as f:
            json.dump(coco_metadata, f)


if __name__=="__main__":
    '''
    vid = cv2.VideoCapture("no-targets-cut.mp4")
    grab_frame = lambda: vid.read()[1]
    '''
    img = cv2.imread("fieldgrab.png")
    grab_frame = lambda: img.copy()


    create_shape_dataset(
        get_frame=grab_frame, 
        shapes_directory="shapes", 
        shape_resolution=36, 
        max_shapes_per_image=3, 
        num_images=10
    )
