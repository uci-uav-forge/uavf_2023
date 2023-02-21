import itertools
import cv2
import numpy as np
from tqdm import tqdm
out_dir = "colors"
for r in tqdm(range(0,256)):
    fname = f"image{r}.png"
    img = np.zeros((256,256,3), dtype=np.uint8)
    img[:,:,0] = r
    for g,b in itertools.product(range(256),range(256)):
        img[g,b,1] = g
        img[g,b,2] = b
    
    # if we want to index the images by H value instead of R value, uncomment this cvtColor line
    # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"{out_dir}/{fname}", img)