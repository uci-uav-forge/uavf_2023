import onnxruntime as ort # pip install onnxruntime
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2 as cv
from typing import List


ort_sess = ort.InferenceSession("shapev1.onnx", providers = ['CPUExecutionProvider']) # For GPU: CUDAExecutionProvider
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

if __name__=="__main__":
    img = Image.open("example-shape.jpg")
    outputs: list[np.ndarray] = make_prediction(img)
    annotated_img = tf.keras.preprocessing.image.img_to_array(img)
    for rect_idx in range(50):
        annotated_img = cv.rectangle(
            img=annotated_img, 
            pt1=np.flip(outputs[0][0][rect_idx][0:2].astype(int)), 
            pt2=np.flip(outputs[0][0][rect_idx][2:].astype(int)), 
            color=(255,0,0))
        print(outputs[1][0], outputs[2][0])
    cv.imshow("Annotated image", annotated_img.astype(np.uint8))
    cv.waitKey(0)