import os
import cv2
import random
from time import time
import argparse
import configparser
import ast
import numpy as np
import pandas as pd

import h5py

import torch
from torch.nn import Dropout, Flatten, Linear, Module, Sequential
from torch.nn import Conv3d, MaxPool3d, MaxUnpool3d
from torch.nn import Conv2d, MaxPool2d, MaxUnpool2d

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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

        self.model = self.create_sequential()

    def create_sequential(self):

        sequential = []

        for (conv3d, dropout, max_pool) in zip(self.conv3d_layers, self.conv3d_dropout_layers, self.conv3d_pool_layers):

            sequential.extend([conv3d, dropout, max_pool])

        for (conv2d, dropout, max_pool) in zip(self.conv2d_layers, self.conv2d_dropout_layers, self.conv2d_pool_layers):

            sequential.extend([conv2d, dropout, max_pool])

        sequential.append(self.flatten)

        for dense in self.dense_layers:

            sequential.append(dense)

        return Sequential(*sequential)

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

        return Flatten()
                
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

        output = self.model(x)

        return output

if __name__ == '__main__':

    EC = Encoder3D('./config/config.cfg')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize([0.0], [1.0]) 
        ])

    images = pd.read_csv('data/fotos_idx.csv', index_col=0)

    

    print(images.columns)

    # TODO: Agregar activation