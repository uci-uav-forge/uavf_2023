from io import TextIOWrapper
import itertools
import math
import os
import random
import json
from math import floor, ceil
from typing import Callable

import cv2
import numpy as np
from image_rotation import get_rotated_image, get_shape_bbox
from text_rendering import drawText, get_shape_text_area

def create_shape_dataset(get_frame: Callable[[], cv2.Mat], 
                         shapes_directory:str, 
                         shape_resolution_fn: Callable[[],int] = lambda: 36, 
                         max_shapes_per_image: int = 3, 
                         num_images:int = 100,
                         blur_radius_fn=lambda: 3,
                         data_split=[1,0,0],
                         output_dir="output",
                         noise_scale=10,
                         seg_size=128
                         ):
    '''
    creates a directory called "output" adjacent to wherever this is run from and fills it with output images.

    ARGUMENTS:

    `get_frame` is a function that returns an image (cv2 mat). This could just return the same image every time, a random image from a directory, or a frame from a video. The implementation doesn't matter as long as the return type is consistent.

    `shapes_directory` contains the images of the shapes to apply to the background.

    `shape_resolution` is the width that shapes will be drawn at, in pixels

    `max_shapes_per_image` is how many shapes are on each image, max. It will choose a number between 0 and this parameter for each output image.
    
    `num_images` is how many images to output

    `blur_radius_fn` is a function that returns the blur radius

    `data_split` is a 3-tuple describing the proportions of train, validation, and test data generated respectively.

    `noise_scale` is the standard deviation for the gaussian noise
    '''
    shapes = dict(
        (
            name.split(".")[0], 
            cv2.imread(f'{shapes_directory}/{name}')
        ) 
        for name in os.listdir(shapes_directory)
    )
    shape_names_and_categories = list(zip(range(1,len(shapes)+1), sorted(shapes.keys())))
    with open("shape_name_labels.json","w") as f:
        json.dump(dict(shape_names_and_categories),f)
    if output_dir not in os.listdir():
        os.mkdir(output_dir)
        for subfolder in ["images","segmentation_masks"]:
            os.mkdir(f"{output_dir}/{subfolder}")
            for split in ["train","validation","test"]:
                os.mkdir(f"{output_dir}/{subfolder}/split")
    
    image_idx=0
    def add_shapes(orig_frame: cv2.Mat, annotations_file: TextIOWrapper, orig_seg_mask: cv2.Mat):
        height, width = orig_frame.shape[:2]
        for _shape_idx in range(random.randint(0,max_shapes_per_image)):
            frame=orig_frame.copy()
            seg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # category_num is in [1,13] and shape_name is in ["circle", "square", ...]
            category_num, shape_name= random.choice(shape_names_and_categories)

            shape_source_h, shape_source_w = shapes[shape_name].shape[:2]
            shape_resize_w = shape_resolution_fn() # the shape source image's width and height
            
            resized_shape = cv2.resize(shapes[shape_name], (shape_resize_w, int(shape_resize_w*shape_source_h/shape_source_w)))
            letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            text_bbox = [[int(dim*shape_resize_w/shape_source_w) for dim in point] for point in get_shape_text_area(shape_name)]
            drawText(
                img=resized_shape, 
                letter=letter, 
                bounding_box=text_bbox,
                color=(0,255,0),
                thickness=max(1,shape_resize_w//20)
                )
            shape_to_draw = get_rotated_image(
                img=resized_shape,
                theta=np.random.random()*np.pi*2
            )
            bbox = get_shape_bbox(shape_to_draw)
            shape_h, shape_w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            
            shape_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            text_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))

            x_offset = random.randint(0,width-shape_w)
            y_offset = random.randint(0, height-shape_h)
            for y, x in itertools.product(range(1,shape_h),range(1,shape_w)):
                pixel_color = shape_to_draw[bbox[0]+x][bbox[1]+y]
                gaussian_noise = np.random.normal(loc=0,scale=noise_scale,size=3)

                if pixel_color[1]>0:
                    frame[y+y_offset][x+x_offset] = text_color+gaussian_noise
                    seg_mask[y+y_offset][x+x_offset] = 125
                elif pixel_color[2]>0 or pixel_color[0]>0:
                    frame[y+y_offset][x+x_offset] = shape_color+gaussian_noise
                    seg_mask[y+y_offset][x+x_offset] = 255

            crop_ymin = y_offset-ceil((seg_size-shape_h)/2)
            crop_ymax =y_offset+floor((seg_size-shape_h)/2)+shape_h
            if crop_ymin<0:
                crop_ymax=seg_size
                crop_ymin=0
            elif crop_ymax>height:
                crop_ymin=height-seg_size
                crop_ymax=height

            crop_xmin = x_offset-ceil((seg_size-shape_w)/2)
            crop_xmax =x_offset+floor((seg_size-shape_w)/2)+shape_w
            if crop_xmin<0:
                crop_xmax=seg_size
                crop_xmin=0
            elif crop_xmax>width:
                crop_xmin=width-seg_size
                crop_xmax=width
            
            blur_radius = blur_radius_fn()
            blur_fn = lambda x: x
            if blur_radius>0:
                blur_fn = lambda x: cv2.blur(x, (blur_radius, blur_radius))
            cv2.imwrite(
                f"./{output_dir}/{split_dir_name}/images/{output_file_name}",
                blur_fn(frame[crop_ymin:crop_ymax,crop_xmin:crop_xmax])
            )
            cv2.imwrite(
                f"./{output_dir}/{split_dir_name}/segmentation_masks/{output_file_name}",
                seg_mask[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
            )
            annotations_file.writelines(["\n",",".join(map(str,[output_file_name,category_num,x_offset,y_offset,x_offset+shape_w,y_offset+shape_h]))])

    for num_in_split, split_dir_name in [(data_split[0]*num_images, "train"),(data_split[1]*num_images, "validation"), (data_split[2]*num_images, "test")]:
        annotations_file = open(f"./{output_dir}/{split_dir_name}annotations.csv","w")
        annotations_file.writelines(["image,label,xmin,ymin,xmax,ymax"])
        for _ in range(math.ceil(num_in_split)):
            image_idx+=1
            frame=get_frame()
            # shape: (height, width, 3) e.g. (2988, 5312, 3)
            output_file_name = f"image{image_idx}.png"
            # seg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            add_shapes(frame, annotations_file, None)
            print("\r[{0}{1}] {2} ({3}%)".format("="*int(image_idx/num_images*20), " "*(20-int(image_idx/num_images*20)), f"Finished {image_idx}/{num_images}", int(image_idx/num_images*100)), end="")
        annotations_file.close()
        print("\nDone!")

if __name__=="__main__":
    '''
    vid = cv2.VideoCapture("no-targets-cut.mp4")
    grab_frame = lambda: vid.read()[1]
    '''
    background_images = []
    for file in os.listdir("backgrounds"):
        background_images.append(cv2.imread(f"backgrounds/{file}"))
    def generate_frame_function(frame_size:int = 512):
        '''
        Returns a function that returns a random window of `img` of size `frame_size`x`frame_size`.
        '''
        def grab_frame():
            img=random.choice(background_images)
            original_h,original_w = img.shape[:2]                
            h = max(frame_size,random.randint(min(frame_size, original_h), max(frame_size,original_h)))
            img=cv2.resize(img, (int(original_w/original_h*h), h))

            h,w = img.shape[:2]
            origin_x = random.randint(0,w-frame_size)
            origin_y = random.randint(0,h-frame_size)
            return img[origin_y:origin_y+frame_size,origin_x:origin_x+frame_size].copy()
        return grab_frame
    

    create_shape_dataset(
        get_frame=generate_frame_function(), 
        shapes_directory="shapes", 
        shape_resolution_fn=lambda: max(10,int(np.random.normal(30,7))), 
        max_shapes_per_image=7, 
        blur_radius_fn=lambda: np.random.randint(3,8),
        num_images=100,
        output_dir="seg_output2",
        data_split=[0.85,0.1,0.05],
        noise_scale=2
    )