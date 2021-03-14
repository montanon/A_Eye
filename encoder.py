import torch.nn as nn
import keras.layers import Conv2D,
                        MaxPooling2D,
                        Dense,
                        Dropout,
                        Input,
                        Flatten,SeparableConv2D
import numpy as np

class EncoderCNN(nn.Module):

    def __init__(self, **kwarg):

        super().__init__()
        self.input_layer = 
        