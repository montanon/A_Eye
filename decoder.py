import argparse
import ast
import configparser
import os
import random
from copy import deepcopy
from time import time

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.nn import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                      Dropout, Linear, MaxPool2d, MaxPool3d, MaxUnpool2d,
                      MaxUnpool3d, Module, Sequential, Unflatten)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from dataloader import Rescale, RGBImageDataset, ToCuda, ToTensor
from encoder import Encoder3D


def ConfigParse(cfg_file, model='DECODER'):
    parser = configparser.ConfigParser()
    parser.read(cfg_file)
    args = parser[model]
    return args


class Decoder3D(Module):

    def __init__(self, cfg):
        super(Decoder3D, self).__init__()
        self.args = ConfigParse(cfg)
        self.conv2d_max_unpool = True if self.args['conv2d_max_unpool'] == 'true' else False

        if self.conv2d_max_unpool:
            self.conv2d_unpool_kernels = ast.literal_eval(
                self.args['conv2d_unpool_kernels'])
            self.conv2d_unpool_layers = [self.create_2d_unpool_layer(
                n_layer) for n_layer in range(int(self.args['n_convs2d']))]
        else:
            self.conv2d_unpool_layers = []

        self.conv2d_in_channels = ast.literal_eval(
            self.args['conv2d_in_channels'])
        self.conv2d_out_channels = ast.literal_eval(
            self.args['conv2d_out_channels'])
        self.conv2d_kernels = ast.literal_eval(self.args['conv2d_kernels'])
        self.conv2d_layers = [self.create_2d_conv_layer(
            n_layer) for n_layer in range(int(self.args['n_convs2d']))]
        self.conv2d_dropout = True if self.args['conv2d_dropout'] == 'true' else False

        if self.conv2d_dropout:
            self.conv2d_dropout_layers = [self.create_dropout_layer(
            ) for n_layer in range(int(self.args['n_convs2d']))]
        else:
            self.conv2d_dropout_layers = []

        self.conv3d_in_channels = ast.literal_eval(
            self.args['conv3d_in_channels'])
        self.conv3d_out_channels = ast.literal_eval(
            self.args['conv3d_out_channels'])
        self.conv3d_kernels = ast.literal_eval(self.args['conv3d_kernels'])
        self.conv3d_layers = [self.create_3d_conv_layer(
            n_layer) for n_layer in range(int(self.args['n_convs3d']))]
        self.conv3d_dropout = True if self.args['conv3d_dropout'] == 'true' else False

        if self.conv3d_dropout:
            self.conv3d_dropout_layers = [self.create_dropout_layer(
            ) for n_layer in range(int(self.args['n_convs3d']))]
        else:
            self.conv3d_dropout_layers = []

        self.conv3d_max_unpool = True if self.args['conv3d_max_unpool'] == 'true' else False

        if self.conv3d_max_unpool:
            self.conv3d_unpool_kernels = ast.literal_eval(
                self.args['conv3d_unpool_kernels'])
            self.conv3d_unpool_layers = [self.create_3d_unpool_layer(
                n_layer) for n_layer in range(int(self.args['n_convs3d']))]
        else:
            self.conv3d_unpool_layers = []

        self.model = self.create_sequential().cuda()

    def create_sequential(self):
        sequential = []
        for (conv2d, dropout, max_unpool) in zip(self.conv2d_layers, self.conv2d_dropout_layers, self.conv2d_unpool_layers):
            sequential.extend([max_unpool, dropout, conv2d])
        for (conv3d, dropout, max_unpool) in zip(self.conv3d_layers, self.conv3d_dropout_layers, self.conv3d_unpool_layers):
            sequential.extend([max_unpool, dropout, conv3d])
        return Sequential(*sequential)

    def create_3d_conv_layer(self, n_layer):
        return ConvTranspose3d(
            in_channels=self.conv3d_in_channels[n_layer],
            out_channels=self.conv3d_out_channels[n_layer],
            kernel_size=self.conv3d_kernels[n_layer],
            padding=int(self.args['padding']),
            stride=int(self.args['stride']),
            bias=True
        )

    def create_3d_unpool_layer(self, n_layer):
        return MaxUnpool3d(
            kernel_size=self.conv3d_unpool_kernels[n_layer],
            padding=int(self.args['padding'])
        )

    def create_2d_conv_layer(self, n_layer):
        return ConvTranspose2d(
            in_channels=self.conv2d_in_channels[n_layer],
            out_channels=self.conv2d_out_channels[n_layer],
            kernel_size=self.conv2d_kernels[n_layer],
            padding=int(self.args['padding']),
            bias=True
        )

    def create_2d_unpool_layer(self, n_layer):
        return MaxUnpool2d(
            kernel_size=self.conv2d_unpool_kernels[n_layer],
            padding=int(self.args['padding'])
        )

    def create_dropout_layer(self):
        return Dropout(
            p=float(self.args['dropout_p'])
        )

    def forward(self, x, idxs):
        unpool = 0
        idxs = idxs[::-1]
        for layer in self.model:
            print(layer)
            if isinstance(layer, MaxUnpool2d) or isinstance(layer, MaxUnpool3d):
                print(idxs[unpool].shape)
                x = layer(x, idxs[unpool])
                unpool += 1
            else:
                x = layer(x)
        return x


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

    x = img_dataset[10]

    output, pool_idxs = EC(x['image'])

    encoded_img = deepcopy(output)
    encoded_img = encoded_img.cpu().numpy()[0, :, :, :]
    encoded_img = np.transpose(encoded_img, (1, 2, 0))
    plt.imshow(encoded_img)
    plt.show()

    DE = Decoder3D('./config/config.cfg')

    output = DE(output, pool_idxs)

    output = output.cpu().numpy()[0, :, :, :]
    output = np.transpose(output, (1, 2, 0))

    x = x['image'].cpu().numpy()[0, :, :, :]
    x = np.transpose(x, (1, 2, 0))

    img = np.hstack([x, output])
    plt.imshow(img)
    plt.show()
