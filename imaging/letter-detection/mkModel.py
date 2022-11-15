from letterDetectionModel import *
from collections import defaultdict
import keras
from keras.callbacks import LearningRateScheduler
import numpy as np


model = keras.Sequential([
    keras.layers.Conv2D(32, (5, 5), input_shape = (128,128,1), activation = 'relu'),
    keras.layers.Conv2D(64, (5, 5), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(128, (5, 5), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    # keras.layers.Conv2D(256, (5, 5), activation = 'relu'),
    # keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64),
    keras.layers.Dense(35, activation = 'softmax')
    
])


epochs = 20
learning_rate = 0.01
decay_rate = 0.5

def exp_decay(epoch):
    lr_new = learning_rate * np.exp(-decay_rate*epoch)
    return lr_new

# learning schedule callback
lr_rate = LearningRateScheduler(exp_decay)


model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                loss=[keras.losses.SparseCategoricalCrossentropy()], 
                metrics="accuracy")
log_dir = "logs/fit/slModel0_18"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(ds_train, epochs=epochs, shuffle=True, callbacks=[lr_rate, tensorboard_callback])

model.save("./trained_model.h5")
