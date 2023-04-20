import cv2
import os
import numpy as np


video_dir = input('INPUT VIDEO DIRECTORY')
write_dir = 'testimages/'

cap = cv2.VideoCapture(video_dir)


if (cap.isOpened()== False):
    print("Error opening video stream or file")
    # Read until video is completed
    
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        #cv2.imshow('Frame',frame)
        cv2.imwrite(f'{write_dir}frame{len(os.listdir(write_dir))}.png', frame)
        # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

        # Break the loop
    else:
        print('ERROR READING')
        #break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
