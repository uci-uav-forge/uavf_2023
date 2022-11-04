from letterDetectionModel import *
from collections import defaultdict
import keras


model = keras.Sequential([
    keras.layers.Conv2D(10, (3, 3), input_shape = (128,128,1), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(20, (5, 5), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(60, activation = 'relu'),
    keras.layers.Dense(60, activation = 'relu'),
    keras.layers.Dense(35, activation = 'relu')
    
])

model.compile(optimizer=tf.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
                metrics="accuracy")
alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
model.fit(ds_train, epochs=1, shuffle=True)
print("Evaluate: ")
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
d = dict()
for i in range(len(predict)):
    x = int(test_labels[i])
    print(predict[i], test_labels[i])
    d[test_file_paths[i]] = (predict[i], test_labels[i])
    print(test_file_paths[i],d[test_file_paths[i]])

alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
def evaluateChars(predictions:dict[tuple[int,int]]):
    for x in predictions.items():
        print(x)
    numCorrect = defaultdict(int)
    totals = defaultdict(int)
    for name, (predicted, correct) in predictions.items():
        numCorrect[correct] = numCorrect[correct]
        totals[correct] += 1
        if(predicted == correct):
            numCorrect[correct] += 1
    print("xxxxx", len(numCorrect), len(totals))
    for letter, (n, t) in enumerate(zip(numCorrect.values(),totals.values())):
        print(alpha[letter], ":", n,"out of", t)
    
evaluateChars(d)

