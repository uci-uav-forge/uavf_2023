import onnxruntime as ort # pip install onnxruntime
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv


ort_sess = ort.InferenceSession("shapev1.onnx", providers = ['CPUExecutionProvider']) # For GPU: CUDAExecutionProvider
def make_prediction(img: np.ndarray):
    model_input = tf.keras.preprocessing.image.img_to_array(img)[np.newaxis,:,:,:]
    outputs = ort_sess.run(None, {"images": model_input.astype(np.uint8)})
    return outputs

if __name__=="__main__":
    img = Image.open("example-shape.jpg")
    outputs: list[np.ndarray] = make_prediction(img)
    annotated_img = tf.keras.preprocessing.image.img_to_array(img)
    for rect_idx in range(5):
        annotated_img = cv.rectangle(
            img=annotated_img, 
            pt1=np.flip(outputs[0][0][rect_idx][0:2].astype(int)), 
            pt2=np.flip(outputs[0][0][rect_idx][2:].astype(int)), 
            color=(255,0,0))
        # print(outputs[1][0][rect_idx], outputs[2][0][rect_idx])
    cv.imshow("Annotated image", annotated_img.astype(np.uint8))
    cv.waitKey(0)
