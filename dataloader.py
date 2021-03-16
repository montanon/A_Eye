import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class ToCuda(object):

    def __call__(self, sample):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image = sample['image'].to(device=device, dtype=torch.float)

        image = image.unsqueeze(0)

        return {'image': image}

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * 1.5, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * 1.5
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

class ToTensor(object):

    def __call__(self, sample):

        image = sample['image']

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image)}    

class RGBImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.root_dir = root_dir

        self.imgs_df = pd.read_csv(os.path.join(self.root_dir, csv_file), index_col=0)

        self.transform = transform

    def __getitem__(self, idx):

        img_name = self.imgs_df['File'].iloc[idx]

        image = io.imread(img_name)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.imgs_df)

    def show_image(self, image):
        plt.imshow(image)
        plt.pause(0.001)

    def show_sample_images(self):

        # TODO: Se puede mejorar el formato del plot

        fig = plt.figure()

        for n_sample, i in enumerate(self.imgs_df.sample(n=4).index):
            sample = self[i]

            print(i, sample['image'].shape)

            ax = plt.subplot(1, 4, n_sample + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            self.show_image(**sample)

        plt.show()

    def show_sample_tensors(self):

        for n_sample, i in enumerate(self.imgs_df.sample(n=4).index):
            sample = self[i]

            print(i, sample['image'].size())

if __name__ == '__main__':

    img_dataset = RGBImageDataset(csv_file='fotos_idx.csv',
                                root_dir='./data'
                                )

    img_dataset.show_sample_images()

    img_transform = transforms.Compose([Rescale(100)])

    img_dataset = RGBImageDataset(csv_file='fotos_idx.csv',
                                root_dir='./data',
                                transform=transforms.Compose([
                                    Rescale(100),
                                    ToTensor(),
                                    ToCuda()
                                ])
                                )

    img_dataset.show_sample_tensors()