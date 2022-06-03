#!/usr/bin/env python
#title           :train.py
#description     :Network for MR-tod
#author          :Zifei Liang
#date            :2022/05/22
#usage           :Network for Deep learing from MRI to TOD
#python_version  :3.6

# import sys
# positionOfPath = 1
# # sys.path.insert(positionOfPath, 'R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn')
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath('R:\zhangj18lab\zhangj18labspace\Zifei_Data\MouseHuman_proj\DeepNet_Learn')
# eng.generate_test_MATLABdwiTODKaffmanKerasLoop0408(nargout = 0)
# eng.quit()

print("No such option")
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.transform
from skimage import data, io, filters
import numpy as np
from numpy import array
import os
from keras.models import load_model
# from scipy.misc import imresize
import argparse

import Utils, Utils_model
from Utils_model import VGG_LOSS

print("No such option")

image_shape = (3,3,3,60)

def test_model(input_hig_res, model, number_of_images, output_dir):
    
    x_test_lr, x_test_hr = Utils.load_test_data_for_model(input_hig_res, '.npy', number_of_images)
    Utils.plot_test_generated_images_for_model(output_dir, model, x_test_hr, x_test_lr)

def test_model_for_lr_images(input_low_res, model, number_of_images, output_dir):

    x_test_lr = Utils.load_test_data(input_low_res, '.npy', number_of_images)
    Utils.plot_test_generated_images(output_dir, model, x_test_lr)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-ihr', '--input_hig_res', action='store', dest='input_hig_res',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/MouseHuman_proj/DeepNet_Learn/',
                    help='Path for input images Hig resolution')
                    
    parser.add_argument('-ilr', '--input_low_res', action='store', dest='input_low_res',
                        default='R:/zhangj18lab/zhangj18labspace/Zifei_Data/MouseHuman_proj/DeepNet_Learn/',
                    help='Path for input images Low resolution')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
                    help='Path for Output images')
    
    # parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/gen_modelTOD500.h5' ,
    #                 help='Path for model')
    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/gen_modelTOD700_28channelKimBatch512.h5' ,
                    help='Path for model')
    # parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/gen_modelTOD1700_28channelKim.h5' ,
    #                 help='Path for model')

    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=25 ,
                    help='Number of Images', type=int)
                    
    parser.add_argument('-t', '--test_type', action='store', dest='test_type', default='test_lr_images',
                    help='Option to test model output or to test low resolution image')
    
    values = parser.parse_args()
    
    loss = VGG_LOSS(image_shape)  
    model = load_model(values.model_dir , custom_objects={'vgg_loss': loss.vgg_loss})
    
    if values.test_type == 'test_model':
        test_model(values.input_hig_res, model, values.number_of_images, values.output_dir)
        
    elif values.test_type == 'test_lr_images':
        test_model_for_lr_images(values.input_low_res, model, values.number_of_images, values.output_dir)
        
    else:
        print("No such option")




