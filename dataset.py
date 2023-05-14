import os
import glob
import re
from io import StringIO
from skimage import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.transforms import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from dataset import *

### Label encoding and decoding

def one_hot_encoding(num_of_labels, labels):
    '''
    num_of_labels: total number of unique labels
    labels: list of integer labels

    Return: encoded labels of length num_of_labels
    '''
    encoded = np.zeros(num_of_labels)
    for label in labels:
        encoded[label - 1] = 1
    return encoded

def decode_labels(labels, thresh):
    '''
    labels: List of the numbers outputed in the encoded format
    thresh: Threshold for the classification probability

    Return: a list of integer labels
    '''
    output = []
    for i in range(len(labels)):
        if labels[i] >= thresh:
            output.append(i + 1)
    return output

### Dataset
class ImageDataset(Dataset):
    
  def __init__(self, label_file, dir, num_labels, transform=None):
    '''
        label_file: the label file to load, ending with .csv
        dir: directory to the folder where the data saved
        num_labels: the number of unique labels
        transform: transform functions to be applied to the image
    '''
    self.dir = dir
    self.file = self.dir + label_file
    self.num_labels = num_labels
    self.transform = transform

    # load the csv file
    with open(self.file) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    self.df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    # for training set, there is an additional column of labels
    if self.df.shape[1] == 3:
        self.train = True
    else:
        self.train = False

    # restrict the number of samples for training and validation
    self.df = self.df.head(10000)

  def __len__(self):
    '''
        Length of the dataset
    '''
    return self.df.shape[0]

  def __getitem__(self, i):
    '''
        Get one sample from the dataset
        i: the index of the sample
    '''
    img_file = self.dir + 'data/' + self.df.iloc[i]['ImageID']
    # load image
    img = io.imread(img_file)
    # apply transform
    if self.transform is not None:
        img = self.transform(img)
    # get the caption
    caption = self.df.iloc[i]['Caption']
    # get the label and encode the label
    if self.train:
        labels = self.df.iloc[i]['Labels']
        labels = labels.split()
        labels = [int(l) for l in labels]
        # one-hot-encoding
        labels = one_hot_encoding(self.num_labels, labels)
        return (img, labels, caption)
    return (img, caption)