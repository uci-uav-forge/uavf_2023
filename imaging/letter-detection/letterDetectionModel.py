import pandas as pd
import tensorflow as tf
from tensorflow import keras

# change train dataset directory here
train_directory = './train/dataset'
# change file name here
train_df = pd.read_csv(train_directory + '/labels.txt')

train_file_paths = train_df['file'].values
train_labels = train_df[' label'].values

ds_train = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
ds_train = ds_train.shuffle(2000, seed=10)



# change test dataset directory here
test_directory = './test/dataset'
# change file name here
test_df = pd.read_csv(test_directory + '/labels.txt')

test_file_paths = test_df['file'].values
test_labels = test_df[' label'].values
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
#conv2d(numFilters, kernelSize)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape = (128,128,1), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(35)
    
])

model.compile(optimizer=tf.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
                metrics="accuracy")

model.fit(ds_train, epochs=2, shuffle=True)
print("Evaluate: ")
result = model.evaluate(ds_test)
