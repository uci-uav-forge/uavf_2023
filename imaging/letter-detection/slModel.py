from letterDetectionModel import *
from collections import defaultdict
import keras
import numpy as np
from keras.callbacks import LearningRateScheduler



model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), input_shape = (128,128,1), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    #keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    #keras.layers.MaxPooling2D(pool_size = (2,2)),
    #keras.layers.Dense(64, activation = 'relu'),
    #keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    #keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    #keras.layers.Dense(256, activation = 'relu'),
    #keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(35, activation = 'softmax')
    
])

# class LearningRateReducerCb(tf.keras.callbacks.Callback):

#   def on_epoch_end(self, epoch, logs={}):
#     old_lr = self.model.optimizer.lr.read_value()
#     new_lr = old_lr * 0.8
#     print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
#     self.model.optimizer.lr.assign(new_lr)

# class LR_Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):

#   def __init__(self, initial_learning_rate):
#     self.initial_learning_rate = initial_learning_rate

#   def __call__(self, step):
#      return self.initial_learning_rate / (step + 1)
#      return return_value.numpy()
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


# class LearningRateLogger(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super().__init__()
#         self._supports_tf_logs = True

#     def on_epoch_end(self, epoch, logs=None):
#         if logs is None or "learning_rate" in logs:
#             return
#         logs["learning_rate"] = self.model.optimizer.lr
log_dir = "logs/fit/slModel0_18"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(ds_train, epochs=epochs, shuffle=True, callbacks=[lr_rate, tensorboard_callback])

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

d = {}
for i in range(len(predict)):
    file_path = test_file_paths[i]
    prediction = predict[i]
    actual_label = test_labels[i]
    d.update({file_path: (prediction, actual_label)})

alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
def evaluateChars(predictions:dict[tuple[int,int]]):
    numCorrect = defaultdict(int)
    totals = defaultdict(int)
    for predicted, correct in predictions.values():
        numCorrect[correct]
        totals[correct] += 1
        if(predicted == correct):
            numCorrect[correct] += 1
    for letter, (n, t) in enumerate(zip(numCorrect.values(),totals.values())):
        print(alpha[letter], ":", n,"out of", t)
    
evaluateChars(d)