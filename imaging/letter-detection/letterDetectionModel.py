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

ds_train = ds_train.map(train_read_image).map(augment).batch(10)
ds_test = ds_test.map(test_read_image).batch(10)
#conv2d(numFilters, kernelSize)
model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), input_shape = (128,128,1), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    #keras.layers.Dense(64, activation = 'relu'),
    #keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    #keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    #keras.layers.Dense(256, activation = 'relu'),
    #keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(35)
    
])

model.compile(optimizer=tf.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
                metrics="accuracy")

model.fit(ds_train, epochs=3, shuffle=True)
print("Evaluate: ")
result = model.evaluate(ds_test)
result = model.predict(ds_test)
# result is a 2D matrix: each row represents each image, and each column represents the "likelihood" for each key
# find the max column value for each row

predict = []
for row in range(len(result)):
    max = 0
    max_col = 0
    i = 0
    for col in range(len(result[0])):
        if result[row][col] > max:
            max = result[row][col]
            max_col = col
    predict.append(max_col)

#wrong_predictions = []
#for i in range(len(predict)):
#    if predict[i] != test_labels[i]:
#        wrong_predictions.append((chr(test_labels[i]), chr(predict[i])))
#print(wrong_predictions)
dict = {}
for i in range(len(predict)):
    dict[test_file_paths[i]] = (predict[i], test_labels[i])

print(dict)


    
