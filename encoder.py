import os
import cv2
import random
from time import time
import argparse
import ast

import h5py
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
import configparser

import matplotlib.pyplot as plt

def ArgsParse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--nconv', type=int, default=3)

    parser.add_argument('--activation', type=str, default='relu')

    parser.add_argument('--padding', type=str, default='same')

    args = parser.parse_args()

    return args

def ConfigParse(cfg_file):

    parser = configparser.ConfigParser()

    parser.read(cfg_file)

    args = parser['ENCODER']

    return args

class Encoder3D(Model):

    def __init__(self, cfg):

        super(Encoder3D, self).__init__()

        self.args = ConfigParse(cfg)

        self.image = Input(
                shape=(
                        int(self.args['image_width']), 
                        int(self.args['image_height']),
                        3,
                        1
                    )
            )

        self.conv3d_features = ast.literal_eval(self.args['conv3d_features'])
        self.conv3d_kernels = ast.literal_eval(self.args['conv3d_kernels'])

        self.conv3d_layers = [self.create_3d_conv_layer(n_layer) for n_layer in range(int(self.args['n_convs3d']))]    

        self.conv3d_max_pool = True if self.args['conv3d_max_pool'] == 'true' else False

        if self.conv3d_max_pool:
            self.conv3d_pool_kernels = ast.literal_eval(self.args['conv3d_pool_kernels'])
            self.conv3d_pool_layers = [self.create_3d_pool_layer(n_layer) for n_layer in range(int(self.args['n_convs3d']))]  
        else:
            self.conv3d_pool_layers = []      

        self.conv2d_features = ast.literal_eval(self.args['conv2d_features'])
        self.conv2d_kernels = ast.literal_eval(self.args['conv2d_kernels'])

        self.conv2d_layers = [self.create_2d_conv_layer(n_layer) for n_layer in range(int(self.args['n_convs2d']))]

        self.conv2d_max_pool = True if self.args['conv2d_max_pool'] == 'true' else False

        if self.conv2d_max_pool:
            self.conv2d_pool_kernels = ast.literal_eval(self.args['conv2d_pool_kernels'])
            self.conv2d_pool_layers = [self.create_2d_pool_layer(n_layer) for n_layer in range(int(self.args['n_convs2d']))]  
        else:
            self.conv2d_pool_layers = []   

        self.flatten = self.create_flatten_layer() 

        self.dense_features = ast.literal_eval(self.args['dense_features'])

        self.dense_layers = [self.create_dense_layer(n_dense) for n_dense in range(int(self.args['n_dense']))]

    def create_3d_conv_layer(self, n_layer):

        return Convolution3D(
                self.conv3d_features[n_layer], 
                self.conv3d_kernels[n_layer],
                activation=self.args['activation'],
                padding=self.args['padding'],
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )
                    
    def create_3d_pool_layer(self, n_layer):

        return MaxPooling3D(
                pool_size=self.conv3d_pool_kernels[n_layer],
                strides=None,
                padding=self.args['padding'],
                data_format=None
            )     
        
    def create_2d_conv_layer(self, n_layer):

        return Convolution2D(
                self.conv2d_features[n_layer],
                self.conv2d_kernels[n_layer],
                activation=self.args['activation'],
                padding=self.args['padding'],
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )  

    def create_2d_pool_layer(self, n_layer):

        return MaxPooling2D(
                pool_size=self.conv2d_pool_kernels[n_layer],
                strides=None,
                padding=self.args['padding'],
                data_format=None
            )     

    def create_flatten_layer(self):

        return Flatten(
            data_format=None
        )
                
    def create_dense_layer(self, n_dense):

        return Dense(
                    self.dense_features[n_dense],
                    activation=None,
                    use_bias=True,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None
                )

        def run(self, x):

            x = self.input(x)

            for conv3d_layer in self.conv3d_layers:
                x = conv3d_layer(x)

            for conv2d_layer in self.conv2d_layers:
                x = conv2d_layer

            x = self.flatten(x)

            for dense_layer in self.dense_layers:
                x = dense_layer

            return x

if __name__ == '__main__':

    EC = Encoder3D('./config/config.cfg')

        