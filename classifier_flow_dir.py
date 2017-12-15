
# coding: utf-8

# # Dog vs. Cat Classifier
# This notebook is supposed to experiment on different aspects of a image classifier: types of layers (fully connected, convolutional), training sample sizes, image generator, pretrained models, etc. 


from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image
import time


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

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/scratch/bnallamo_armis/wsliu/dogs_cats/train',  # this is the target directory
        target_size=(200, 200),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/scratch/bnallamo_armis/wsliu/dogs_cats/data/train',
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='binary')


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

start = time.time()
print("==== Start model training ====")
model.fit_generator(
        train_generator,
        steps_per_epoch=23000 // batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=2000 // batch_size);
run_time = time.time() - start
print("==== End model training ====")
f = open('run_times.txt', 'a')
f.write("""
======== Begin of a new test ========
In this test we use cpu: 1 node with 16 cores and 32 Gb mem. 
We use a flow from directory to do data augmentation on the fly.  
We use 23000 training samples, 2000 testing samples and 30 epochs. 
Image size is set to 200*200. 
""")
f.write("Time used for model training: %.2f\n" % run_time)
f.close()

model.save_weights('naive_cnn_full_sample.h5')  # always save your weights after training or during training

