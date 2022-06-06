#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4
# Modified by Zifei Liang  2022/06/06

from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.models import Sequential
from scipy import signal
from scipy import ndimage
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Converting my comment into an answer. Given two angles a (gt), b (prediction) as radians you get the angle difference by
#
# tf.atan2(tf.sin(a - b), tf.cos(a - b))
# By definition tf.atan2 gives the difference automatically in the closed interval [-pi, +pi] (that is, [-180 degrees, +180 degrees]).
#
# Hence, you can use
#
# tf.reduce_mean(tf.abs(tf.atan2(tf.sin(a - b), tf.cos(a - b))))
# I think Keras understand this TensorFlow code.

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        return K.mean(K.square(y_true-y_pred))

        # img_input = Input(shape=(128, 128, 1), name='grayscale_input')
        # x = Conv2D(3, (3, 3), padding='same', name='grayscale_RGBlayer')(img_input)
        # net_in = Model(img_input, x)
        # vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
        # part_vgg19 = Sequential(vgg19.layers[:-1])
        # for l in vgg19.layers:
        #     l.trainable = False
        # output = part_vgg19(net_in.outputs)
        # model = Model(inputs=net_in.inputs, outputs=output)
        # return K.mean(K.square(model(y_true) - model(y_pred)))
        #



        #
        # vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        # vgg19.trainable = False
        # # Make trainable as False
        # for l in vgg19.layers:
        #     l.trainable = False
        # model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        # model.trainable = False
        #
        # return K.mean(K.square(model(y_true) - model(y_pred)))


def angular_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    cosine = K.sum(y_true * y_pred, axis=-1)
    return 2 * tf.math.acos(cosine) / np.pi

# model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001),
#             metrics=[angular_distance])
    
def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam
