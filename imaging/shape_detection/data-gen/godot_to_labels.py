# takes dataset with masks generated with godot and turns it into a dataset with labels for YOLOv8
from yolo_seg_main import get_polygon
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def preprocess_img(img):
    # blur image with random kernel size
    kernel_size = 3 + 2*np.random.randint(0, 2)
    if np.random.randint(0,2)==0:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    else:
        img = cv2.boxFilter(img, -1, (kernel_size, kernel_size), normalize=False)
    # add random noise with random variance
    variance = np.random.randint(0, 10)
    img = img + np.random.normal(0, variance, img.shape)
    # clamp values to 0-255
    img = np.clip(img, 0, 255)
    return img


if __name__ == "__main__":
    datagen_dir = os.path.dirname(os.path.abspath(__file__))
    categories_to_shapes = json.load(open(f"{datagen_dir}/shape_name_labels.json","r"))
    shapes_to_categories = {shape:category for category, shape in categories_to_shapes.items()}
    input_dir = "/tmp/godot_data"
    output_dir = f"{datagen_dir}/data"
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        os.makedirs(f"{output_dir}/labels/{split_name}", exist_ok=True)
        os.makedirs(f"{output_dir}/images/{split_name}", exist_ok=True)
    num_images = len(os.listdir(f"{input_dir}/images"))
    for num in tqdm(range(num_images)):
        if int(num)<0.85*num_images:
            split_name = "train"
        elif int(num)<0.95*num_images:
            split_name = "validation"
        else:
            split_name = "test" 
        img = cv2.imread(f"{input_dir}/images/image{num}.png")
        if img is None:
            tqdm.write(f"image read error for {img}")
            continue
        img = preprocess_img(img)
        cv2.imwrite(f"{output_dir}/images/{split_name}/image{num}.png", img)
        f = open(f"{output_dir}/labels/{split_name}/image{num}.txt", "w")
        for mask_file_name in os.listdir(f"{input_dir}/masks/{num}"):
            mask_path = f"{input_dir}/masks/{num}/{mask_file_name}"
            shape_name = mask_file_name.split("_")[0] 
            mask = cv2.imread(mask_path)
            polygon = get_polygon(mask)

            if len(polygon) <= 2:
                tqdm.write(f"no polygon found for {mask_path}")
                continue
            normalized_polygon = polygon / np.array([mask.shape[1], mask.shape[0]])
            f.write(f"{shapes_to_categories[shape_name]} {' '.join(map(str, normalized_polygon.flatten()))}\n")
        f.close()

    