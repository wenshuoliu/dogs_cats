
# coding: utf-8

# This file is dog/cat classifier built by fine-tuning a pretrained VGG16 net. 

import os
from keras.layers.core import Dense
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator

path = "/scratch/bnallamo_armis/wsliu/dogs_cats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
batch_size=64

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model

gen = ImageDataGenerator()
train_gen = gen.flow_from_directory(path+'train', target_size=(224, 224), class_mode='categorical', shuffle=True,
                                    batch_size=batch_size)
test_gen = gen.flow_from_directory(path+'data/train', target_size=(224, 224), class_mode='categorical', shuffle=False,
                                   batch_size=batch_size)

model.pop() #defined earlier model = vgg.model
for layer in model.layers: layer.trainable=False
model.add(Dense(2, activation='softmax'))

opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_gen, steps_per_epoch=train_gen.n // batch_size, epochs=5, validation_data=test_gen, 
                    validation_steps = test_gen.n // batch_size)

model.save_weights(model_path+'fine_tune_all.h5')
