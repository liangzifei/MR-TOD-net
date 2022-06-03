#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator and Discriminator)
#author          :Deepak Birla
#date            :2018/10/30
#usage           :from Network import Generator, Discriminator
#python_version  :3.5.4 

# Modules
from keras.layers import Dense, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, UpSampling3D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU, ReLU
from keras.layers import add
from keras.backend import squeeze
# from keras.backend.tensorflow_backend import AtrousConvolution3D

def res_block_gen_noBN(model, kernal_size, filters, strides):

    gen = model

    model = Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    # model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    # model = BatchNormalization(momentum = 0.5)(model)

    model = add([gen, model])

    return model
# Residual block
def res_block_noBN(model):
    gen = model

    model = Dense(1024, activation='relu')(model)
    # model = BatchNormalization(momentum=0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Dense(1024, activation='relu')(model)
    # model = BatchNormalization(momentum=0.5)(model)
    model = add([gen, model])
    # model = Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum=0.5)(model)
    # # Using Parametric ReLU
    # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    # model = Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum=0.5)(model)
    #
    # model = add([gen, model])
    return model

# Residual block
def res_block_gen(model, kernal_size, filters, strides):

    gen = model

    model = Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = add([gen, model])

    return model


# Residual block
def res_block(model):
    gen = model

    model = Dense(1024, activation='relu')(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Dense(1024, activation='relu')(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = add([gen, model])
    # model = Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum=0.5)(model)
    # # Using Parametric ReLU
    # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    # model = Conv3D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum=0.5)(model)
    #
    # model = add([gen, model])
    return model
    
def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling3D(size = (2,2,1))(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape

    def generator(self):
        gen_input = Input(shape=self.noise_shape)
        # model = Conv3D(filters=1024, kernel_size=2, strides=1, padding="same", activation='relu')(gen_input)
        # for index in range(4):
        #     model = res_block_gen_noBN(model, 3, 1024, 1)
        #     model = Dropout(0.5)(model)

        model = Conv3D(filters=2048, kernel_size=2, strides=1, padding="valid", activation='relu')(gen_input)

        model = BatchNormalization(momentum=0.5)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,shared_axes=[1, 2])(model)
        model = Dropout(0.4)(model)
        # model = ReLU(alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
        model = Conv3D(filters=1024, kernel_size=2, strides=1, padding="valid", activation='relu')(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)
        model = Dropout(0.4)(model)
        # model = Conv3D(filters=512, kernel_size=2, strides=1, padding="valid", activation='relu')(model)
        # model = Conv3D(filters=512, kernel_size=2, strides=1, padding="valid", activation='relu')(model)
        # model = ReLU(alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        #     model)
        # model = Dense(2048, activation='relu')(model)
        # model = Dense(1024, activation='relu')(model)
        # model = Dense(1024, activation='relu')(model)
        # model = Dense(1024, activation='relu')(model)
        # for index in range(4):
        #     model = res_block_noBN(model)
        #     model = Dropout(0.4)(model)

        # model = Dense(2048, activation='relu')(model)
        # model = BatchNormalization(momentum=0.5)(model)
        # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        #     model)
        # model = Dropout(0.4)(model)
        #
        # model = Dense(2048, activation='relu')(model)
        # model = BatchNormalization(momentum=0.5)(model)
        # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        #     model)
        # model = Dropout(0.4)(model)

        model = Dense(1024, activation='relu')(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
            model)
        model = Dropout(0.4)(model)

        # model = Dense(1024, activation='relu')(model)
        model = Dense(512, activation='relu')(model)
        model = Dense(64, activation='relu')(model)
        model = Dense(28)(model)
        generator_model = Model(inputs=gen_input, outputs=model)
        return generator_model
	    # gen_input = Input(shape = self.noise_shape)
	    # model = Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(gen_input)
	    # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	    # gen_model = model
        #
        # # Using 16 Residual Blocks
	    # for index in range(4):
	    #     model = res_block_gen(model, 3, 64, 1)
	    # model = Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
	    # model = BatchNormalization(momentum = 0.5)(model)
	    # model = add([gen_model, model])
	    # # Using 2 UpSampling Blocks
	    # # for index in range(1):
	    # #     model = up_sampling_block(model, 3, 256, 1)
	    # model = Conv3D(filters = 1, kernel_size = 3, strides = 1, padding = "same")(model)
	    # generator_model = Model(inputs = gen_input, outputs = model)
	    # return generator_model

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model
