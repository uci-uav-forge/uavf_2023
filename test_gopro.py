#!/usr/bin/python3
from imaging.camera import GoProCamera
import cv2 as cv

cam = GoProCamera()
img = cam.get_image()
cv.imwrite("test_gopro_img.png", img)