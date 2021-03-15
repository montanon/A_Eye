import os
import cv2
import random
from time import time
import argparse
import configparser
import ast

import h5py

import torch
from torch.nn import Dropout, Linear, Module
from torch.nn import Conv3d, MaxPool3d, MaxUnpool3d
from torch.nn import Conv2d, MaxPool2d, MaxUnpool2d

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

class Encoder3D(Module):

    def __init__(self, cfg):

        super(Encoder3D, self).__init__()

        self.args = ConfigParse(cfg)


        self.conv3d_in_channels = ast.literal_eval(self.args['conv3d_in_channels'])
        self.conv3d_out_channels = ast.literal_eval(self.args['conv3d_out_channels'])

        self.conv3d_kernels = ast.literal_eval(self.args['conv3d_kernels'])

        self.conv3d_layers = [self.create_3d_conv_layer(n_layer) for n_layer in range(int(self.args['n_convs3d']))]   

        self.conv3d_dropout = True if self.args['conv3d_dropout'] == 'true' else False

        if self.conv3d_dropout:
            self.conv3d_dropout_layers = [self.create_dropout_layer() for n_layer in range(int(self.args['n_convs3d']))]
        else:
            self.conv3d_dropout_layers = []

        self.conv3d_max_pool = True if self.args['conv3d_max_pool'] == 'true' else False

        if self.conv3d_max_pool:
            self.conv3d_pool_kernels = ast.literal_eval(self.args['conv3d_pool_kernels'])
            self.conv3d_pool_layers = [self.create_3d_pool_layer(n_layer) for n_layer in range(int(self.args['n_convs3d']))]  
        else:
            self.conv3d_pool_layers = []  


        self.conv2d_in_channels = ast.literal_eval(self.args['conv2d_in_channels'])
        self.conv2d_out_channels = ast.literal_eval(self.args['conv2d_out_channels'])

        self.conv2d_kernels = ast.literal_eval(self.args['conv2d_kernels'])

        self.conv2d_layers = [self.create_2d_conv_layer(n_layer) for n_layer in range(int(self.args['n_convs2d']))]

        self.conv2d_dropout = True if self.args['conv2d_dropout'] == 'true' else False

        if self.conv2d_dropout:
            self.conv2d_dropout_layers = [self.create_dropout_layer() for n_layer in range(int(self.args['n_convs2d']))]
        else:
            self.conv2d_dropout_layers = []

        self.conv2d_max_pool = True if self.args['conv2d_max_pool'] == 'true' else False

        if self.conv2d_max_pool:
            self.conv2d_pool_kernels = ast.literal_eval(self.args['conv2d_pool_kernels'])
            self.conv2d_pool_layers = [self.create_2d_pool_layer(n_layer) for n_layer in range(int(self.args['n_convs2d']))]  
        else:
            self.conv2d_pool_layers = []   


        self.flatten = self.create_flatten_layer() 


        self.dense_in_features = ast.literal_eval(self.args['dense_in_channels'])
        self.dense_out_features = ast.literal_eval(self.args['dense_out_channels'])

        self.dense_layers = [self.create_connected_layer(n_dense) for n_dense in range(int(self.args['n_dense']))]

    def create_3d_conv_layer(self, n_layer):

        return Conv3d(
                in_channels=self.conv3d_in_channels[n_layer],
                out_channels=self.conv3d_out_channels[n_layer],
                kernel_size=self.conv3d_kernels[n_layer],
                padding=int(self.args['padding']),
                bias=True              
            )
                    
    def create_3d_pool_layer(self, n_layer):

        return MaxPool3d(
                kernel_size=self.conv3d_pool_kernels[n_layer],
                padding=int(self.args['padding'])
            )     
        
    def create_2d_conv_layer(self, n_layer):

        return Conv2d(
                in_channels=self.conv2d_in_channels[n_layer],
                out_channels=self.conv2d_out_channels[n_layer],
                kernel_size=self.conv2d_kernels[n_layer],
                padding=int(self.args['padding']),
                bias=True              
            )

    def create_2d_pool_layer(self, n_layer):

        return MaxPool2d(
                kernel_size=self.conv2d_pool_kernels[n_layer],
                padding=int(self.args['padding'])
            )     

    def create_flatten_layer(self):

        return torch.flatten
                
    def create_connected_layer(self, n_layer):

        return Linear(
                    in_features=self.dense_in_features[n_layer],
                    out_features=self.dense_out_features[n_layer],
                    bias=True
                )

    def create_dropout_layer(self):

        return Dropout(
            p=float(self.args['dropout_p'])
        )

    def forward(self, x):

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

        