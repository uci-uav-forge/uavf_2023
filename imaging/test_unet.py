import tensorflow as tf
from keras.utils import normalize
import numpy as np
import cv2 as cv
model = tf.keras.models.load_model('/home/holden/code/uavf_2023/imaging/colordetect/unet-rgb.hdf5')

def _get_seg_masks(images: np.ndarray) -> np.ndarray:
    '''
    images is of shape (batch_size, 128, 128) 
    '''
    # model_input = np.expand_dims(images, axis=3)
    model_input = normalize(images, axis=1)
    prediction_raw = model.predict(model_input)
    prediction = np.argmax(prediction_raw, axis=3)
    return prediction

img = cv.imread('/home/holden/code/uavf_2023/flight_data/02-24|20:14:11/shape_seg0/crop0.png')
img = cv.resize(img,(128,128))

masks = _get_seg_masks(np.array([img])).astype(np.uint8) # 0=background, 1=shape, 2=letter

cv.imwrite('source.png', img)
cv.imwrite("res.png", masks[0]*127)
