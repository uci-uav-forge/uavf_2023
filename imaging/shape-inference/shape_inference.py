import onnxruntime as ort # pip install onnxruntime
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2 as cv
from typing import List

file_prefix = "/home/holden/code/uavf_2023/imaging/shape-inference/"

ort_sess = ort.InferenceSession(file_prefix+"shapev1.onnx", providers = ['CPUExecutionProvider']) # For GPU: CUDAExecutionProvider
def make_prediction(img: np.ndarray) -> List[np.ndarray]:
    '''
    Takes an image of any shape [x,y,3].
    Returns a list of numpy arrays formatted like this:
    e.g. `print([a.shape for a in make_prediction(img)])`:
    `[
        (1, 100, 4),
        (1, 100),
        (1, 100),
        (1,)
    ]`
    The last array just one int, the number of bounding boxes returned.
    The first array is n x 4, where n is the number of bounding boxes, and it has the rectangle coordinates as [min_y, min_x, max_y, max_x]
    For example, to render this rectangle in opencv, you would use
    ```
    cv.rectangle(
            img=annotated_img, 
            pt1=np.flip(outputs[0][0][rect_idx][0:2].astype(int)), 
            pt2=np.flip(outputs[0][0][rect_idx][2:].astype(int)), 
            color=(255,0,0))
    ```
    The second array, n x 4, is the confidences for each bounding box.
    The third array, n x 4, is the class label. For example, 0 might be circle, and 1 might be square, etc. These aren't necessarily the actual labels though. Ask whoever trained the most recent shape model for what the labels are if they haven't documented it anywhere.
    '''
    model_input = tf.keras.preprocessing.image.img_to_array(img)[np.newaxis,:,:,:]
    outputs = ort_sess.run(None, {"images": model_input.astype(np.uint8)})
    return outputs

def supress_nonmax_bboxes(bbox_list: np.ndarray, iou_threshold):
    '''
    bbox_list should be like this:
    [
        [x1,y1,x2,y2]
    ]
    and sorted by confidence, with the highest confidence boxes first in the list
    x1 and y1 are the top left, and x2 and y2 are the bottom right
    
    iou threshold should be a float between 0 and 1.
    if the intersection-over-union (iou) of two boxes is > threshold, it'll discard the one with lower confidence.
    '''
    correct_bboxes = []
    for x1,y1,x2,y2 in bbox_list:
        is_duplicate = False
        for x3,y3,x4,y4 in correct_bboxes:
            if not (x3>x2 or x1>x4 or y3>y2 or y1>y4):# if they overlap
                # gets the coordinates of the intersection
                intersection_x1, intersection_x2 = sorted([x1,x2,x3,x4])[1:3]
                intersection_y1, intersection_y2 = sorted([y1,y2,y3,y4])[1:3]
                intersection_area = (intersection_x2-intersection_x1)*(intersection_y2-intersection_y1)
                a1 = (x2-x1)*(y2-y1)
                a2 = (x4-x3)*(y4-y3)
                iou = intersection_area/(a1+a2-intersection_area)
                if iou >= iou_threshold:
                    is_duplicate=True
        if not is_duplicate:
            correct_bboxes.append(np.array([x1,y1,x2,y2]))
    return correct_bboxes

if __name__=="__main__":
    img = Image.open(file_prefix+"example-shape.jpg")
    outputs: list[np.ndarray] = make_prediction(img)
    bboxes = supress_nonmax_bboxes(outputs[0][0], 0)
    annotated_img = tf.keras.preprocessing.image.img_to_array(img)
    for rect_idx in range(50):
        annotated_img = cv.rectangle(
            img=annotated_img, 
            pt1=np.flip(outputs[0][0][rect_idx][0:2].astype(int)), 
            pt2=np.flip(outputs[0][0][rect_idx][2:].astype(int)), 
            color=(255,0,0))
        print(outputs[1][0], outputs[2][0])
    # cv.imshow("Annotated image", annotated_img.astype(np.uint8))
    # cv.waitKey(0)
