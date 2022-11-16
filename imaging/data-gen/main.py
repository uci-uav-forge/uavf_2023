# import the opencv library
import cv2
import os
import random

# define a video capture object
# def create_shape_dataset(video_file_name:str, shapes_directory:str, shape_resolution: int = 36)
vid = cv2.VideoCapture("no-targets-cut.mp4")
shapes = dict(
    (name.split(".")[0], cv2.imread(f'shapes/{name}')) 
    for name in os.listdir("shapes")
)

while True:
    ret, frame = vid.read()
    height, width = frame.shape[:2]
    # shape: (height, width, 3) e.g. (2988, 5312, 3)
    shape_name = random.choice(list(shapes.keys()))
    shape_true_h, shape_true_w = shapes[shape_name].shape[:2]
    shape_resize_w = 36
    shape = cv2.resize(shapes[shape_name], (shape_resize_w, int(shape_resize_w*shape_true_h/shape_true_w)))
    shape_h, shape_w = shape.shape[:2]
    for _test_idx in range(random.randint(0,3)):
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        x_offset = random.randint(0,width-shape_w)
        y_offset = random.randint(0, height-shape_h)
        for y in range(shape_h):
            for x in range(shape_w):
                if shape[y][x][2] == 255:
                    frame[y+y_offset][x+x_offset] = color

    cv2.imshow('frame', cv2.resize(frame, (1600, 900)))
    if cv2.waitKey(0) == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()