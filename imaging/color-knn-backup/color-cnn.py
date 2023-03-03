import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IAMGES= True


CLASS_NAMES = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'gray', 'white', 'black', 'brown']

test_datagen, train_datagen, valid_datagen = ImageDataGenerator(rescale = (1./255)),  \
                              ImageDataGenerator(rescale = (1./255)), \
                                ImageDataGenerator(rescale = (1./255))

training_set = train_datagen.flow_from_directory(r'Dataset\train', target_size = (1,1), batch_size = 64, \
                                                    classes = CLASS_NAMES, class_mode = 'categorical', )

test_set = test_datagen.flow_from_directory(r'Dataset\test', target_size = (1,1), batch_size = 32,  \
                                                classes = CLASS_NAMES, class_mode = 'categorical',  )

valid_set = valid_datagen.flow_from_directory(r'Dataset\val', target_size = (1,1), batch_size = 32,  \
                                                classes = CLASS_NAMES, class_mode = 'categorical',   )


#training_set = 4 dimensional array: num of images, image dimension and then channel so could feed it a one pixel image. 
# want the lost to be low and the accuracy to be high;


model = tf.keras.models.Sequential  ([
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(120,activation = 'relu'), 
        tf.keras.layers.Dense(10,activation = 'softmax')])


#so before getting test data test on training data to see accuracy and adjust epoch to higher if it's at 1 percent

opt = RMSprop(learning_rate = 0.0001)
model.compile(loss='categorical_crossentropy',
		optimizer = opt, metrics = ['accuracy'])

history = model.fit(training_set,
			steps_per_epoch = 250,
			epochs = 50,
			validation_steps = 30,
      validation_data = valid_set)

test_loss,test_accuracy = model.evaluate(test_set,verbose=1)
print("test Loss:", test_loss)
print("test Accuracy :", test_accuracy)


