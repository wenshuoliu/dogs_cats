
#!usr/bin/python

# # Dog vs. Cat Classifier
# This notebook is supposed to experiment on different aspects of a image classifier: types of layers (fully connected, convolutional), training sample sizes, image generator, pretrained models, etc. 


from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Prepare the data, using the .flow_from_directory() method. 

batch_size = 32
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/dogs_cats/'

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        shear_range=0.1, 
        zoom_range=0.1,
        rotation_range=20,
        horizontal_flip=True, 
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        path+'wrong_label',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path+'data/train2000',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


# ## Model specification

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model training:

start = time.time()
print("==== Start model training ====")
model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size, 
        verbose=2);
run_time = time.time() - start
print("==== End model training ====")
f = open('run_times.txt', 'a')
f.write("""
======== Begin of a new test ========
In this test we use gpu: 1 node with 1 gpu and 8G mem. 
We use a flow from directory to do data augmentation on the fly.  
We use 23000 training samples, 2000 testing samples and 50 epochs. 
Image size is set to 224*224. 
""")
f.write("Time used for model training: %.2f\n" % run_time)
f.close()

model.save_weights(path+'models/covmax4_hidden1_wronglabel.h5')  # always save your weights after training or during training

