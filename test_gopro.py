#!/usr/bin/python3
from imaging.camera import GoProCamera
import cv2 as cv
import os
import time

dir_name = f"gopro_tests/{time.strftime(r'%m-%d|%H_%M_%S')}"
os.makedirs(dir_name, exist_ok=True)
cam = GoProCamera()
index = 0
while 1:
    img = cam.get_image()
    print(img.shape)
    cv.imwrite(f"{dir_name}/img{index}.png", img)
    index+=1