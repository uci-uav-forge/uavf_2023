import pandas as pd
import tensorflow as tf
from tensorflow import keras

train_directory = '../letter_generation/train/'
train_df = pd.read_csv(train_directory + 'train.txt')

train_file_paths = train_df['file_name'].values
train_labels = train_df['label'].values

ds_train = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))

test_directory = '../letter_generation/test/'
test_df = pd.read_csv(test_directory + 'test.txt')

test_file_paths = test_df['file_name'].values
test_labels = test_df['label'].values

ds_test = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))

for epoch in range(4):
    for x, y in ds_train:
        # train here
        pass

def train_read_image(image_file, label):
    image = tf.io.read_file(train_directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)  #channel=1 means grayscale
    return image, label

def test_read_image(image_file, label):
    image = tf.io.read_file(test_directory + image_file)
    image = tf.image.decode_image(image, channels=1, dtype=tf.float32)  #channel=1 means grayscale
    return image, label

def augment (image, label):
    image = tf.image.random_brightness(image, max_delta=0.05)
    return image, label

ds_train = ds_train.map(train_read_image).map(augment).batch(2)
ds_test = ds_test.map(test_read_image).batch(2)

model = keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 1)),
    # Convolutional layer: generates 16 filters and multiply each of them across the image.
    #       Then, each epoch will figure out which filters gave the best signals to help match the images to their labels
    # Max pooling compresses the image and enhances the features (each 2x2 block is compressed into 1 block)
    tf.keras.layers.MaxPooling2D(2, 2),
    # Stacking convolutional layers on top of each other to break down the image and learn from abstract features
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # Dropout: improve the efficiency of the neural network by randomly throwing away some of the neurons
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(36)   # 36 is the number of classes
])

model.compile(optimizer=tf.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
                metrics="accuracy")

model.fit(ds_train, epochs=4, shuffle=True)
print("Evaluate: ")
result = model.evaluate(ds_test)
