import math
import os
import random
import json
import itertools
from typing import Callable, Dict, List, Tuple
from io import TextIOWrapper

import cv2
import numpy as np
from image_rotation import get_rotated_image, get_shape_bbox
from text_rendering import drawText, get_shape_text_area

def get_polygon(shape_img: cv2.Mat) -> np.ndarray:
    '''
    Returns the enclosing polygon of the shape in the image. The polygon is a list of points, each point being a list of 2 coordinates.
    '''
    im = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
    im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array(contours[0]).reshape(-1,2)

def add_shapes(frame: cv2.Mat, annotations_file: TextIOWrapper, max_shapes_per_image: int, shape_resolution_fn: Callable[...,int], shapes: Dict[str, cv2.Mat], shape_names_and_categories: List[Tuple[int, str]], noise_scale: int):
        height, width = frame.shape[:2]
        for _shape_idx in range(random.randint(0,max_shapes_per_image)):
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
            polygon = get_polygon(shape_to_draw[bbox[0]:bbox[2], bbox[1]:bbox[3]])
            shape_h, shape_w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            
            shape_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            text_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))

            x_offset = random.randint(0,width-shape_w)
            y_offset = random.randint(0, height-shape_h)
            for y, x in itertools.product(range(1,shape_h), range(1, shape_w)):
                pixel_color = shape_to_draw[bbox[0]+x][bbox[1]+y]
                if pixel_color[1]>0:
                    gaussian_noise = np.random.normal(loc=0,scale=noise_scale,size=3)
                    frame[y+y_offset][x+x_offset] = text_color+gaussian_noise
                elif pixel_color[2]>0 or pixel_color[0]>0:
                    gaussian_noise = np.random.normal(loc=0,scale=noise_scale,size=3)
                    frame[y+y_offset][x+x_offset] = shape_color+gaussian_noise

            transformed_polygon = np.array([[(point[1]+x_offset)/width,(point[0]+y_offset)/height] for point in polygon], dtype=np.float32).reshape(-1)
            if any(transformed_polygon>1):
                raise ValueError
            
            annotations_file.write(f"{category_num} {' '.join(transformed_polygon.astype(str))}\n")


def create_shape_dataset(get_frame: Callable[[], cv2.Mat], 
                         shapes_directory:str, 
                         shape_resolution_fn: Callable[[],int] = lambda: 36, 
                         max_shapes_per_image: int = 3, 
                         num_images:int = 100,
                         blur_radius_fn=lambda: 3,
                         data_split=[1,0,0],
                         output_dir="output",
                         noise_scale=10):
    '''
    creates a directory called "output" adjacent to wherever this is run from and fills it with output images.

    ARGUMENTS:

    `get_frame` is a function that returns an image (cv2 mat). This could just return the same image every time, a random image from a directory, or a frame from a video. The implementation doesn't matter as long as the return type is consistent.

    `shapes_directory` contains the images of the shapes to apply to the background.

    `shape_resolution` is the width that shapes will be drawn at, in pixels

    `max_shapes_per_image` is how many shapes are on each image, max. It will choose a number between 0 and this parameter for each output image.
    
    `num_images` is how many images to output

    `blur_radius` is the size of the gaussian kernel. The default is 3, which means a 3x3 kernel.

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
    shape_names_and_categories = list(zip(range(len(shapes)), sorted(shapes.keys())))
    with open("shape_name_labels.json","w") as f:
        json.dump(dict(shape_names_and_categories),f)
    if output_dir not in os.listdir():
        os.mkdir(output_dir)
        for subfolder in ["images","labels"]:
            os.mkdir(f"{output_dir}/{subfolder}")
            for split in ["train","validation","test"]:
                os.mkdir(f"{output_dir}/{subfolder}/{split}")
    image_idx=0
    
    for num_in_split, split_dir_name in [(data_split[0]*num_images, "train"),(data_split[1]*num_images, "validation"), (data_split[2]*num_images, "test")]:
        for _ in range(math.ceil(num_in_split)):
            image_idx+=1
            frame=get_frame()
            # shape: (height, width, 3) e.g. (2988, 5312, 3)
            output_file_name = f"image{image_idx}"
            with open(f"./{output_dir}/labels/{split_dir_name}/{output_file_name}.txt","w+") as annotations_file:
                add_shapes(frame, annotations_file, max_shapes_per_image, shape_resolution_fn, shapes, shape_names_and_categories, noise_scale)
                blur_radius =blur_radius_fn()
                if blur_radius>0:
                    frame=cv2.blur(frame, (blur_radius, blur_radius))
                cv2.imwrite(f"./{output_dir}/images/{split_dir_name}/{output_file_name}.png", frame)

                print("\r[{0}{1}] {2} ({3}%)".format("="*int(image_idx/num_images*20), " "*(20-int(image_idx/num_images*20)), f"Finished {image_idx}/{num_images}", int(image_idx/num_images*100)), end="")
if __name__=="__main__":
    '''
    vid = cv2.VideoCapture("no-targets-cut.mp4")
    grab_frame = lambda: vid.read()[1]
    '''
    background_images = []
    for file in os.listdir("backgrounds"):
        background_images.append(cv2.imread(f"backgrounds/{file}"))
    def generate_frame_function(frame_size:int = 640):
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
        num_images=10_000,
        output_dir="output",
        data_split=[0.85,0.1,0.05],
        noise_scale=2
    )
