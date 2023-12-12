import argparse
import ast
import configparser
import os
import random
from time import time

import cv2
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input
from keras.layers.convolutional import (
    Convolution2D,
    Convolution3D,
    MaxPooling3D,
    UpSampling3D,
)
from keras.models import Model


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
    args = parser['DEFAULT']
    return args


class Encoder3D:

    def __init__(self, cfg):
        self.args = ConfigParse(cfg)
        self.input = Input(
            shape=(
                int(self.args['input_width']),
                int(self.args['input_height']),
                3,
                1)
        )
        self.conv3d_features = ast.literal_eval(self.args['conv3d_features'])
        self.conv3d_kernels = ast.literal_eval(self.args['conv3d_kernels'])
        self.conv3d_layers = [self.create_3d_conv_layer(
            n_conv) for n_conv in range(int(self.args['n_convs3d']))]
        self.conv2d_features = ast.literal_eval(self.args['conv2d_features'])
        self.conv2d_kernels = ast.literal_eval(self.args['conv2d_kernels'])
        self.conv2d_layers = [self.create_2d_conv_layer(
            n_conv) for n_conv in range(int(self.args['n_convs2d']))]
        self.dense_features = ast.literal_eval(self.args['dense_features'])
        self.dense_layers = [self.create_dense_layer(
            n_dense) for n_dense in range(int(self.args['n_dense']))]

    def create_3d_conv_layer(self, n_conv):
        return Convolution3D(
            self.conv3d_features[n_conv],
            self.conv3d_kernels[n_conv],
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

    def create_2d_conv_layer(self, n_conv):
        return Convolution2D(
            self.conv2d_features[n_conv],
            self.conv2d_kernels[n_conv],
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


if __name__ == '__main__':
    EC = Encoder3D('./config/config.cfg')
