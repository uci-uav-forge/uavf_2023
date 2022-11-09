from collections import defaultdict
import tensorflow as tf
import pandas as pd
import keras


# change test dataset directory here
test_directory = './test/dataset'
# change file name here
test_df = pd.read_csv(test_directory + '/labels.txt')

def test_read_image(image_file, label):
    image = tf.io.read_file(test_directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)  #channel=1 means grayscale
    return image, label

test_file_paths = test_df['file'].values
test_labels = test_df[' label'].values
ds_test = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
ds_test = ds_test.map(test_read_image).batch(2)
model = tf.keras.models.load_model('trained_model.h5')
model.summary()
model.evaluate(ds_test)