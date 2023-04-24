import cv2 as cv
import numpy as np
# model = YOLO('./trained_models/seg-v8n-18epoch.pt')
img_num=0
img = cv.imread(f"../shape_detection/data-gen/data/images/train/image{img_num}.png")
with open(f"../shape_detection/data-gen/data/labels/train/image{img_num}.txt", "r") as f:

    for label in f.readlines():
        label, *poly_str = label.split(' ')
        poly_normalized = np.array([float(x) for x in poly_str]).reshape(-1, 2)
        poly = [(poly_normalized*640).astype(int)]

        # draw polygon
        blank_image = np.zeros((640, 640, 3), np.uint8)
        cv.polylines(img, poly, True, (255, 0, 0), thickness=5)

cv.imshow('image', img)
while cv.waitKey(0) not in [27, ord('q')]:
    pass