from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
import cv2 as cv
import numpy as np
model = YOLO('./trained_models/seg-v8n.pt')
img_num=2
img = cv.imread(f"../shape_detection/data-gen/output/images/train/image{img_num}.png")

res: Results  = model.predict(img)

masks = [m.to("cpu").numpy()*255 for m in res[0].masks.masks]

combined_mask = np.zeros_like(masks[0])
for m in masks:
    combined_mask = np.maximum(combined_mask, m)
cv.imwrite('img.png', img)
cv.imwrite('mask.png', combined_mask)