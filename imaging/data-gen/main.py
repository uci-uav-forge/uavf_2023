import os
import json
import random
import cv2
import numpy as np
from  image_rotation import get_rotated_image, get_shape_bbox

def create_shape_dataset(get_frame, shapes_directory:str, shape_resolution: int = 36, max_shapes_per_image: int = 3, num_images:int = 5000):
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
    annotations = []
    images = []
    for image_idx in range(num_images):
        ret, frame = vid.read()
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
            annotations.append({
                "id": image_idx*max_shapes_per_image+shape_idx,
                "image_id": image_idx,
                "category_id": category_num,
                "bbox": [(x_offset),(y_offset),shape_w,shape_h],
                "area": shape_w*shape_h,
                "segmentation": [],
                "iscrowd": 0
            })
        output_file_name = f"./output/image{image_idx}.png"
        cv2.imwrite(output_file_name, frame)
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
    with open("output/coco.json", "x") as f:
        json.dump(coco_metadata, f)

'''
vid = cv2.VideoCapture("no-targets-cut.mp4")
grab_frame = lambda: vid.read()[1]
'''

img = cv2.imread("fieldgrab.png")
grab_frame = lambda: img.copy()


create_shape_dataset(img, "shapes", 36, 3, 10)
