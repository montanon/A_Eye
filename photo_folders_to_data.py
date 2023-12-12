import os
import shutil
import sys

import cv2
import pandas as pd
import torch
from tqdm import tqdm


def get_photo_files(photos_path, non_folders):
    photo_files = []
    for root, dirs, files in os.walk(photos_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.JPG'):
                file_path = os.path.join(root, filename)
                if file_path.split('/')[3] not in set(non_folders):
                    if file_path.split('/')[3].split('.')[-1] not in set(['jpg', 'JPG']):
                        photo_files.append(file_path)
    return photo_files


def get_photo_hash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def get_photo_hashes(photo_files):
    photo_hashes = {}
    with tqdm(total=len(photo_files), file=sys.stdout) as pbar:
        for photo_file in photo_files:
            photo = cv2.imread(photo_file)
            img_hash = get_photo_hash(photo)
            if not os.path.exists('./data/fotos/{}.JPG'.format(img_hash)):
                photo_hash = photo_hashes.get(img_hash, [])
                photo_hash.append(photo_file)
                photo_hashes[img_hash] = photo_hash
            pbar.update(1)
    return photo_hashes


def move_imgs_to_data(photo_hashes):
    for photo_hash in photo_hashes.items():
        photo_file = photo_hash[1][0]
        photo_id = str(photo_hash[0])
        photo_name = photo_id + '.JPG'
        if os.path.exists('./data/fotos'):
            if not os.path.exists('./data/fotos/{}'.format(photo_name)):
                shutil.copy(photo_file, './data/fotos/{}'.format(photo_name))


if __name__ == '__main__':
    photos_path = './Fotos'
    non_folders = [
        'Folder1',
        'Folder2',
        'Folder3'
    ]
    photo_files = get_photo_files(photos_path, non_folders)
    photo_hashes = get_photo_hashes(photo_files)
    move_imgs_to_data(photo_hashes)
