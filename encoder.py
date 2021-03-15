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

from dataloader import RGBImageDataset, Rescale, ToTensor,ToCuda

import matplotlib.pyplot as plt

def ConfigParse(cfg_file, model='ENCODER'):

    parser = configparser.ConfigParser()

    parser.read(cfg_file)

    args = parser[model]

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

        self.model = self.create_sequential().cuda()

    def create_sequential(self):

        sequential = []

        for (conv3d, dropout, max_pool) in zip(self.conv3d_layers, self.conv3d_dropout_layers, self.conv3d_pool_layers):

            sequential.extend([conv3d, dropout, max_pool])

        for (conv2d, dropout, max_pool) in zip(self.conv2d_layers, self.conv2d_dropout_layers, self.conv2d_pool_layers):

            sequential.extend([conv2d, dropout, max_pool])

        return Sequential(*sequential)

    def create_3d_conv_layer(self, n_layer):

        return Conv3d(
                in_channels=self.conv3d_in_channels[n_layer],
                out_channels=self.conv3d_out_channels[n_layer],
                kernel_size=self.conv3d_kernels[n_layer],
                padding=int(self.args['padding']),
                stride=int(self.args['stride']),
                bias=True              
            )
                    
    def create_3d_pool_layer(self, n_layer):

        return MaxPool3d(
                kernel_size=self.conv3d_pool_kernels[n_layer],
                padding=int(self.args['padding']),
                return_indices=True
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
                padding=int(self.args['padding']),
                return_indices=True
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

        idxs = []

        for layer in self.model:

            if isinstance(layer, MaxPool2d) or isinstance(layer, MaxPool3d):
                x, idx = layer(x)
                idxs.append(idx)

            else:
                x = layer(x)

        return x, idxs

if __name__ == '__main__':

    EC = Encoder3D('./config/config.cfg')

    img_dataset = RGBImageDataset(csv_file='fotos_idx.csv',
                                root_dir='./data',
                                transform=transforms.Compose([
                                    Rescale(100),
                                    ToTensor(),
                                    ToCuda()
                                ])
                                )

    #img_dataset.show_sample_tensors()

    x = img_dataset[0]

    output, pool_idxs = EC(x['image'])

    print(output.shape)

    # TODO: Agregar activation