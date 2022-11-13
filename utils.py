import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import random
import math
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
import cv2


class ImageDataset(Dataset):

    def __init__(self, labels, image_path, image_size, single_channel=True, random_transform=True):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_path = image_path
        self.image_size = image_size
        self.single_channel = single_channel
        self.random_transform = random_transform
        self.normalize = transforms.Compose(
            [   
                # todo
                #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize([0.5], [0.5]),
            ])
        self.transform_with_random = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize(size=self.image_size),
                transforms.RandomCrop(size=self.image_size),
            ])
        
        self.transform_without_random = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Resize(size=self.image_size),
                transforms.CenterCrop(size=self.image_size),
            ])

    def get_image(self, path):
        image = Image.open(path)
        if self.single_channel:
            image = image.convert('L')
        if self.random_transform:
            image = self.transform_with_random(image)
        else:
            image = self.transform_without_random(image)
        image = self.normalize(image)
        return image

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = self.get_image(self.image_path[idx])
        label = self.labels[idx]
        return image, label

        
def create_dataloaders(dataset_train, dataset_test, params):
    temp_params = params.copy()
    shuffle_train = temp_params.pop('shuffle_train')
    shuffle_test = temp_params.pop('shuffle_test')
    dataloader_train = DataLoader(dataset_train, shuffle=shuffle_train, **temp_params)
    dataloader_test = DataLoader(dataset_test, shuffle=shuffle_test, **temp_params)
    return dataloader_train, dataloader_test


def get_dataloaders_imagedata(df_train, df_test, image_size, dataloader_params, single_channel=True, random_transform_train=True, random_transform_test=False):
    dataset_train = ImageDataset(df_train.label_cat.values, df_train.image_path.values, image_size, single_channel, random_transform_train)
    dataset_test = ImageDataset(df_test.label_cat.values, df_test.image_path.values, image_size, single_channel, random_transform_test)
    return create_dataloaders(dataset_train, dataset_test, dataloader_params)


def get_dataloaders_clf(X_train, Y_train, X_test, Y_test, dataloader_params):
    dataset_train = TensorDataset(X_train, Y_train)
    dataset_test = TensorDataset(X_test, Y_test)
    return create_dataloaders(dataset_train, dataset_test, dataloader_params)


def get_BUSI_dataset(data_path, drop_normals=False, make_balance=False,random_state=-1, normal_vs_cancer=False):
    image_path = []
    labels = []
    if make_balance and random_state == -1:
        print('Random state has not been passed')
    for label in os.listdir(data_path):
        for image in os.listdir(f'{data_path}/{label}'):
            image_path.append(f'{data_path}/{label}/{image}')
            labels.append(label)

    df = pd.DataFrame({'image_path': image_path, 'label': labels})
    if normal_vs_cancer:
        df['label'] = df['label'].apply(lambda x: x if x=='normal' else 'cancer') 
    df['label_cat'] = df['label'].astype('category').cat.codes
    df = df[~df.image_path.str.contains('mask')].copy()
    if make_balance:
        g = df.groupby('label')
        df = g.apply(lambda x: x.sample(g.size().min(), random_state=random_state).reset_index(drop=True)).reset_index(drop=True)
    return df


def get_chestxray_dataset(data_path, make_balance=False,random_state=-1):
    data_split = ['train','val','test']
    df = {}
    if make_balance and random_state == -1:
        print('Random state has not been passed')
    for ds in data_split:
        image_path = []
        labels = []
        for label in os.listdir(f'{data_path}/{ds}'):
            if label[0] == '.':
                continue
            for image in os.listdir(f'{data_path}/{ds}/{label}'):
                if image[0] == '.':
                    continue
                image_path.append(f'{data_path}/{ds}/{label}/{image}')
                labels.append(label)

        df[ds] = pd.DataFrame({'image_path': image_path, 'label': labels})
        df[ds]['label_cat'] = df[ds]['label'].astype('category').cat.codes
    if make_balance:
        g = df['train'].groupby('label')
        df['train'] = g.apply(lambda x: x.sample(g.size().min(), random_state=random_state).reset_index(drop=True)).reset_index(drop=True)

    return df['train']


def get_MRI_dataset(data_path, make_balance=False, random_state=-1):
    mask_files = glob(f'{data_path}/kaggle_3m/*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]

    def label(mask):
        value = np.max(cv2.imread(mask))
        return 'abnormal' if value > 0 else 'normal'
    df = pd.DataFrame({"image_path": image_files,
                       "mask_path": mask_files,
                       "label":[label(x) for x in mask_files]})

    df['label_cat'] = df['label'].astype('category').cat.codes
    if make_balance:
        g = df.groupby('label')
        df = g.apply(lambda x: x.sample(g.size().min(), random_state=random_state).reset_index(drop=True)).reset_index(drop=True)
    return df


def get_figure(*imgs):
    from skimage import color

    imgs = [img.squeeze().detach().cpu() for img in imgs] 
    imgs = [color.rgb2gray(img.permute(1, 2, 0).numpy()) if img.shape[0]==3 else img.numpy() for img in imgs]
        
    fig = plt.figure(figsize=(15, 5))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(imgs[i])
    return fig


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def dice_score(pred, targs):
    non_prob = (pred>0.5).float()
    dice = 2. * (non_prob*targs).sum(dim=(1,2,3)) / (non_prob+targs).sum(dim=(1,2,3))
    dice[torch.isinf(dice)] = 0.
    dice[torch.isnan(dice)] = 0.
    return dice


def correct_list(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct
