
#!usr/bin/python

# # Dog vs. Cat Classifier
# This notebook is supposed to experiment on different aspects of a image classifier: types of layers (fully connected, convolutional), training sample sizes, image generator, pretrained models, etc. 


from keras.preprocessing.image import ImageDataGenerator
#from PIL import Image
import time
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Prepare the data, using the .flow_from_directory() method. 

batch_size = 64
path = '/nfs/turbo/intmed-bnallamo-turbo/wsliu/Data/dogs_cats/'
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2,
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
        path+'HD/train/',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path+'HD/validate',
        target_size=(256, 256),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')


# ## Model specification

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import pickle


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=model_path+'bench2000_0130_2.h5', verbose=0, save_best_only=True, save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1.e-8)

earlystop = EarlyStopping(monitor='val_loss', patience=50)

# Model training:

start = time.time()
print("==== Start model training ====")
history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size, 
        callbacks = [checkpointer, reduce_lr, earlystop],
        verbose=2);
run_time = time.time() - start
print("==== End model training ====")

with open('output/bench2000_0130_2.pkl', 'wb') as f:

        pickle.dump(history.history, f, -1)

#model.save_weights(path+'models/covmax4_hidden1_wronglabel.h5')  # always save your weights after training or during training

