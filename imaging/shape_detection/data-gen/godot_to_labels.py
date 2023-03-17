# takes dataset with masks generated with godot and turns it into a dataset with labels for YOLOv8
from yolo_seg_main import get_polygon
import os
import cv2
import json
import numpy as np

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    categories_to_shapes = json.load(open(f"{base_dir}/shape_name_labels.json","r"))
    shapes_to_categories = {shape:category for category, shape in categories_to_shapes.items()}
    output_dir = f"{base_dir}/labels"
    os.makedirs(output_dir, exist_ok=True)
    for dir in os.listdir(f"{base_dir}/masks"): # for all the subdirectories
        with open(f"{base_dir}/labels/image{dir}.txt", "w") as f:
            for mask_file_name in os.listdir(f"{base_dir}/masks/{dir}"):
                mask_path = f"{base_dir}/masks/{dir}/{mask_file_name}"
                shape_name = mask_file_name.split("_")[0] 
                mask = cv2.imread(mask_path)
                polygon = get_polygon(mask)

                if len(polygon) == 0:
                    print(f"no polygon found for {mask_path}")
                    continue
                normalized_polygon = polygon / np.array([mask.shape[1], mask.shape[0]])
                f.write(f"{shapes_to_categories[shape_name]} {' '.join(map(str, normalized_polygon.astype(str).flatten()))}")

    