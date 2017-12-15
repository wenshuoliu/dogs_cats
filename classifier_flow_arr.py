
# coding: utf-8

# # Dog vs. Cat Classifier
# This notebook is supposed to experiment on different aspects of a image classifier: types of layers (fully connected, convolutional), training sample sizes, image generator, pretrained models, etc. 

# ## Image Generator tools from _keras_

from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image
import time
#from keras.utils import np_utils

# Prepare the data, using the .flow_from_directory() method. 

batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.1, 
        zoom_range=0.2, 
        horizontal_flip=True, 
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this generator is used to read all image into a large numpy array
read_datagen = ImageDataGenerator()

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
f = open('run_times.txt', 'a')
f.write("""
======== Begin of a new test ========
In this test we use cpu: 1 node with 16 cores and 32 Gb mem. 
We read the whole data into memory first, and then train model. 
We use 23000 training samples, 2000 testing samples and 30 epochs. 
Image size is set to 200*200. 
""")

print("==== Reading data into memory ====")
batch = 0
start = time.time()
for X_train, y_train in read_datagen.flow_from_directory(
        '/scratch/bnallamo_armis/wsliu/dogs_cats/train',  # this is the target directory
        target_size=(200, 200),  # all images will be resized to 150x150
        batch_size=23000,
        class_mode='binary'):  # since we use binary_crossentropy loss, we need binary labels
    batch += 1
    if batch >= 1:
        break

# this is a similar generator, for validation data
batch = 0
for X_test, y_test in  read_datagen.flow_from_directory(
        '/scratch/bnallamo_armis/wsliu/dogs_cats/data/train',
        target_size=(200, 200),
        batch_size=2000,
        class_mode='binary'):
    batch += 1
    if batch >= 1:
        break
run_time = time.time() - start
print("Time used for reading data: ", run_time)
f.write("Time used for reading data: %.2f\n" % run_time)
f.close()

# Y_train = np_utils.to_categorical(y_train, 2)
# Y_test = np_utils.to_categorical(y_test, 2)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# ## Model specification

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Model training:
print("==== Start model training ====")
start = time.time()

model.fit_generator(
        train_generator,
        steps_per_epoch=23000 // batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=2000 // batch_size);

run_time = time.time() - start
print("==== End model training ====")
print("Time used for model training: ", run_time)
f = open('run_times.txt', 'a')
f.write("Time used for model training: %.2f\n" % run_time)
f.close()

# Save the model in h5 file
model.save_weights('naive_cnn_full_sample.h5')  # save the weights after training or during training

