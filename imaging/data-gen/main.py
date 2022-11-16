import cv2
import os
import json
import random

def create_shape_dataset(video_file_name:str, shapes_directory:str, shape_resolution: int = 36, max_shapes_per_image: int = 3, num_images:int = 5000):
    vid = cv2.VideoCapture(video_file_name)
    shapes = dict(
        (name.split(".")[0], cv2.imread(f'{shapes_directory}/{name}')) 
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
    for i in range(num_images):
        ret, frame = vid.read()
        height, width = frame.shape[:2]
        # shape: (height, width, 3) e.g. (2988, 5312, 3)
        for test_idx in range(random.randint(0,max_shapes_per_image)):
            shape_name, category_num = random.choice(list(zip(shapes.keys(), range(len(shapes)))))
            shape_true_h, shape_true_w = shapes[shape_name].shape[:2]
            shape_resize_w = shape_resolution
            shape = cv2.resize(shapes[shape_name], (shape_resize_w, int(shape_resize_w*shape_true_h/shape_true_w)))
            shape_h, shape_w = shape.shape[:2]
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            x_offset = random.randint(0,width-shape_w)
            y_offset = random.randint(0, height-shape_h)
            for y in range(shape_h):
                for x in range(shape_w):
                    if shape[y][x][2] == 255:
                        frame[y+y_offset][x+x_offset] = color
            annotations.append({
                "id": i+5000*(test_idx+1),
                "image_id": i,
                "category_id": category_num,
                "bbox": [(x_offset+shape_w//2),(y_offset+shape_h//2),shape_w,shape_h],
                "area": shape_w*shape_h,
                "segmentation": [],
                "iscrowd": 0
            })
        output_file_name = f"./output/shape{i}.png"
        cv2.imwrite(output_file_name, frame)
        images.append({
            "id": i,
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
    # After the loop release the cap object
    vid.release()

create_shape_dataset("no-targets-cut.mp4", "shapes", 36, 3, 10)