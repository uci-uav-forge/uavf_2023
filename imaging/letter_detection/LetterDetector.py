import tensorflow as tf
import numpy as np

class LetterDetector:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)
        self.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

    def predict(self, img):
        result = self.model.predict(img)
        # max = 0
        # max_col = 0
        # for col in range(len(result[0])):
        #     if result[0][col] > max:
        #         max = result[0][col]
        #         max_col = col
        return result
        # print(f"Max: {self.labels[max_col]}: {max}")


if __name__ == "__main__":
    model = LetterDetector('../trained_model.h5')
    print(model.model.summary())
    image = tf.keras.utils.load_img('./test/dataset/data/0.jpg', color_mode = "grayscale")
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    print(input_arr.shape)
    # model.predict(input_arr)
    # ds_test = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
    # ds_test = ds_test.map(test_read_image).batch(2)
    # model.predict(ds_test)