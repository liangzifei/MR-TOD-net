#!/usr/bin/env python
#title           :train.py
#description     :Network for MR-tod
#author          :Zifei Liang
#date            :2022/05/22
#usage           :Network for Deep learing from MRI to TOD
#python_version  :3.6

from Network import Generator
import Utils_model, Utils
from Utils_model import VGG_LOSS
from keras.models import load_model
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import argparse

np.random.seed(10)
# Remember to change image shape if you are having different size of images
image_shape = (3,3,3, 60)
dis_shape = (1,1,1, 28)


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, input_dir, tgt_dir, output_dir, model_save_dir, number_of_images, train_test_ratio):
    
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = Utils.load_training_data(input_dir, tgt_dir, '.npy', number_of_images, train_test_ratio)
    loss = VGG_LOSS(image_shape)
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0], image_shape[1], image_shape[2], image_shape[3])
    generator = Generator(shape).generator()
    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)

    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            gan_loss = generator.train_on_batch(image_batch_lr, image_batch_hr)

        # print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        loss_file = open(model_save_dir + 'losses.txt' , 'a')
        loss_file.write('epoch%d : gan_loss = %s ; \n' %(e, gan_loss) )
        loss_file.close()

        if e == 1 or e % 5 == 0:
            rand_nums = np.random.randint(0, x_test_hr.shape[0], size=batch_size)
            image_batch_hr = x_test_hr[rand_nums]
            image_batch_lr = x_test_lr[rand_nums]
            test_loss = generator.test_on_batch(image_batch_lr, image_batch_hr)
            print("test_loss :", test_loss)
            test_loss = str(test_loss)
            loss_file = open(model_save_dir + 'test_losses.txt', 'a')
            loss_file.write('epoch%d : test_loss = %s ; \n' % (e, test_loss))
            loss_file.close()
            # Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 100 == 0:
            generator.save(model_save_dir + 'gen_modelTOD%d.h5' % e)
            # discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir',
                        default='./Matlab-CODE/' ,
                    help='Path for input images')

    parser.add_argument('-tgt', '--tgt_dir', action='store', dest='tgt_dir',
                        default='./Matlab-CODE/' ,
                        help='Path for input images')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    
    parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/' ,
                    help='Path for model')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=512,
                    help='Batch Size', type=int)
                    
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=3000,
                    help='number of iteratios for trainig', type=int)
                    
    parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=870000,
                    help='Number of Images', type= int)
                    
    parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.95,
                    help='Ratio of train and test Images', type=float)
    
    values = parser.parse_args()
    
    train(values.epochs, values.batch_size, values.input_dir, values.tgt_dir, values.output_dir, values.model_save_dir, values.number_of_images, values.train_test_ratio)


