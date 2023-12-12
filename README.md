# A_Eye

## Overview
Repository created to handle the Encoder-Decoder pair creation and implementation for compressing and predicting the color channels distributions of picture taken by me.

## **Please Note**
Repository is very old and better techniques could easily be applied with Stable Difussion models.

## Files Description

### 'dataloader.py'
#### Classes: 
ToCuda, Rescale, ToTensor, RGBImageDataset — Handle various preprocessing steps like scaling, tensor conversion, and CUDA optimization.
#### Functions: 
Key methods for dataset handling and image visualization (show_image, show_sample_images, show_sample_tensors).

### 'decoder.py'
#### Class:
Decoder3D — Implements a 3D decoder for the autoencoder model.
### Functions:
Includes methods for creating sequential layers, 3D and 2D convolution and unpooling layers, and a forward pass method.

### 'encoder.py'
#### Class:
Encoder3D — Defines a 3D encoder for the autoencoder model.
#### Functions:
Methods for constructing different layers like 3D and 2D convolution and pooling layers, connected and flatten layers, and the forward pass.

### 'photo_folders_to_data.py'
#### Functions:
Scripts for processing photo folders, including hashing and moving images (get_photo_files, get_photo_hash, get_photo_hashes, move_imgs_to_data).

### 'autoencoder_setup.sh'
Shell script for setting up the autoencoder environment, including dependencies installation and environment configuration.
