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
model = tf.keras.models.load_model('trained_model')
model.summary()
# model.evaluate(ds_test)

result = model.predict(ds_test)

predict = []
total = 0
for row in range(len(result)):
    total += 1
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
totalCorrect = 0
alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
def evaluateChars():
    global totalCorrect, total
    numCorrect = defaultdict(int)
    totals = defaultdict(int)
    for i in range(len(predict)):
        # print(f'Guessed {alpha[predict[i]]} and was {alpha[test_labels[i]]}')
        totals[test_labels[i]] += 1
        numCorrect[test_labels[i]] # init key if not created yet
        if(test_labels[i] == predict[i]):
            totalCorrect += 1
            numCorrect[test_labels[i]] += 1
    for letter, (n, t) in enumerate(zip(numCorrect.values(),totals.values())):
        print(alpha[letter], ":", n,"out of", t)
    print(f"Accuracy: {totalCorrect / total}")
    
evaluateChars()

