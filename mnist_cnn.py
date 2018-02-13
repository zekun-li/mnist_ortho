'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import os
gpu_id = '5'
os.environ["THEANO_FLAGS"] = "device=gpu%s,floatX=float32" % gpu_id

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from orthdense import OrthDense
from ortho import Ortho

K.set_image_data_format('channels_first')

batch_size = 128
num_classes = 10
epochs =1 
#epochs = 50
save_dir = os.path.join(os.getcwd(), 'saved_models')       
model_name = 'keras_mnist_trained_model.h5'
if_visualize = False

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

if if_visualize == True:
    import cv2
    cv2.imwrite('raw.png', x_train[0])

#print ('x_train[0]', x_train[0])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape) # (60000,28,28,1)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
if K.image_data_format() == 'channels_first':
    model.add(Ortho(axis = 1))
else:
    model.add(Ortho(axis = -1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Ortho())
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))
#model.add(OrthDense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Save model and weights                                                                                                                               
if not os.path.isdir(save_dir):                                                     
    os.makedirs(save_dir)                                                                         
model_path = os.path.join(save_dir, model_name)                                                                                                                                                            
model.save(model_path)                                                                                                                                                                                     
print('Saved trained model at %s ' % model_path) 


