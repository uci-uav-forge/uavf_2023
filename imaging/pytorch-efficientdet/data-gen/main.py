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
                         data_split=[1,0,0]):
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
    annotations_file = open("./output/annotations.csv","w")
    annotations_file.writelines(["image,label,xmin,ymin,xmax,ymax"])
    for num_in_split, split_dir_name in [(data_split[0]*num_images, "train"),(data_split[1]*num_images, "validation"), (data_split[2]*num_images, "test")]:
        annotations = []
        images = []
        for _ in range(math.ceil(num_in_split)):
            image_idx+=1
            frame=get_frame()
            height, width = frame.shape[:2]
            # shape: (height, width, 3) e.g. (2988, 5312, 3)
            output_file_name = f"image{image_idx}.png"
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
                annotations_file.writelines(["\n",",".join(map(str,[output_file_name,category_num,x_offset,y_offset,x_offset+shape_w,y_offset+shape_h]))])
            frame=cv2.blur(frame, (blur_radius, blur_radius))
            cv2.imwrite(f"./output/{split_dir_name}/{output_file_name}", frame)
            print("\r[{0}{1}] {2} ({3}%)".format("="*int(image_idx/num_images*20), " "*(20-int(image_idx/num_images*20)), f"Finished {image_idx}/{num_images}", int(image_idx/num_images*100)), end="")
            # print(f"Finished {image_idx}/{num_images}")
    annotations_file.close()

if __name__=="__main__":
    '''
    vid = cv2.VideoCapture("no-targets-cut.mp4")
    grab_frame = lambda: vid.read()[1]
    '''
    img = cv2.imread("fieldgrab.png")
    def grab_frame(frame_size: int = 512):
        # return img.copy()
        h,w = img.shape[:2]
        origin_x = random.randint(0,w-frame_size)
        origin_y = random.randint(0,h-frame_size)
        return img[origin_y:origin_y+frame_size,origin_x:origin_x+frame_size].copy()


    create_shape_dataset(
        get_frame=grab_frame, 
        shapes_directory="shapes", 
        shape_resolution=36, 
        max_shapes_per_image=3, 
        blur_radius=5,
        num_images=10000
    )
