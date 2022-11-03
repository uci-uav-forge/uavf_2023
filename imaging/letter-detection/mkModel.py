# from letterDetectionModel import *
from collections import defaultdict
# import keras


# model = keras.Sequential([
#     keras.layers.Conv2D(10, (3, 3), input_shape = (128,128,1), activation = 'relu'),
#     keras.layers.MaxPooling2D(pool_size = (2,2)),
#     keras.layers.Conv2D(20, (5, 5), activation = 'relu'),
#     keras.layers.MaxPooling2D(pool_size = (2,2)),
#     keras.layers.Dropout(0.1),
#     keras.layers.Flatten(),
#     keras.layers.Dense(60, activation = 'relu'),
#     keras.layers.Dense(60, activation = 'relu'),
#     keras.layers.Dense(35, activation = 'relu')
    
# ])

# model.compile(optimizer=tf.optimizers.Adam(),
#                 loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),], 
#                 metrics="accuracy")
# alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
# model.fit(ds_train, epochs=1, shuffle=True)
# print("Evaluate: ")
# result = model.predict(ds_test)
# predictions = dict()
# for index, x in enumerate(result):
#     maxNum = 0
#     maxInd = 0
#     for pred, y in enumerate(x):
#         if maxNum < y:
#             maxInd = pred
#             maxNum = y
#             print(y, ", ", end = "")
#     predictions[index] = (alpha[test_labels[index]], alpha[maxInd])
# for pred in predictions.items():
#     print(predictions)

x = dict()
x[0] = (1,1)
x[1] = (2,2)
x[2] = (1,3)
x[3] = (1,1)
x[4] = (1,2)
x[5] = (1,3)
x[6] = (1,1)
x[7] = (1,2)
x[7] = (1,3)

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
    
evaluateChars(x)

