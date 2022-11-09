from letterDetectionModel import *
from collections import defaultdict
import keras


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

model.compile(optimizer=tf.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
                metrics="accuracy")
alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
model.fit(ds_train, epochs=1, shuffle=True)
model.save("./trained_model.h5")
model.evaluate(ds_test)

result = model.predict(ds_test)

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

alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
def evaluateChars():
    numCorrect = defaultdict(int)
    totals = defaultdict(int)
    for i in range(len(predict)):
        print(f'Guessed {alpha[predict[i]]} and was {alpha[test_labels[i]]}')
        totals[test_labels[i]] += 1
        numCorrect[test_labels[i]] # init key if not created yet
        if(test_labels[i] == predict[i]):
            numCorrect[test_labels[i]] += 1
    for letter, (n, t) in enumerate(zip(numCorrect.values(),totals.values())):
        print(alpha[letter], ":", n,"out of", t)
    
evaluateChars()

