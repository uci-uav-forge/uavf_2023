import tensorflow as tf
import numpy as np
from PIL import Image
image = Image.open('./test/dataset/data/0.jpg')
class LetterDetector:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)
        self.model.summary()
        self.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

    def predict(self, img):
        image = tf.keras.utils.load_img(img, color_mode = "grayscale")
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])
        result = self.model.predict(input_arr)
        print(result)
        max = 0
        max_col = 0
        for col in range(len(result[0])):
            if result[0][col] > max:
                max = result[0][col]
                max_col = col
        print(f"Max: {self.labels[max_col]}: {max}")


if __name__ == "__main__":
    model = LetterDetector('trained_model.h5')
    model.predict('./test/dataset/data/0.jpg')
    # ds_test = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
    # ds_test = ds_test.map(test_read_image).batch(2)
    # model.predict(ds_test)