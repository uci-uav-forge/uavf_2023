from ultralytics import YOLO
import cv2 as cv
import numpy as np
# model = YOLO('./trained_models/seg-v8n-18epoch.pt')
img_num=2
img = cv.imread(f"../shape_detection/data-gen/output/images/train/image{img_num}.png")
with open(f"../shape_detection/data-gen/output/labels/train/image{img_num}.txt", "r") as f:

    for label in f.readlines():
        label, *poly_str = label.split(' ')
        poly_normalized = np.array([float(x) for x in poly_str]).reshape(-1, 2)
        poly = [(poly_normalized*640).astype(int)]
        img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        # draw polygon
        cv.fillPoly(img, poly, (0, 255, 100, 80))

cv.imwrite('test.png', img)