import re
from io import StringIO
from skimage import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataset import *
from transformers import BertTokenizer

# BERT learned and adapted from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

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
    # ensure hte labels are in a list
    labels = list(labels)
    # decode the labels using the threshold
    output = []
    for i in range(len(labels)):
        if labels[i] >= thresh:
            output.append(i + 1)

    # if there is no label above the provided threshold
    # return one label with the highest probability in the encoded label
    if len(output) == 0:
        max_index = labels.index(max(labels))
        output.append(max_index+1)
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
    
    # tokenizer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # # restrict the number of samples for code test
    # self.df = self.df.head(20)

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
    img_id = self.df.iloc[i]['ImageID']
    img_file = self.dir + 'data/' + img_id
    # load image
    img = io.imread(img_file)
    # apply transform
    if self.transform is not None:
        img = self.transform(img)
    # get the caption
    caption = self.df.iloc[i]['Caption']
    # tokenize
    bert_input = self.tokenizer(caption,padding='max_length', 
                           max_length = 20, 
                           truncation=True, 
                           return_tensors="pt")
    # get the id and mask for caption
    txt_ids = bert_input['input_ids']
    txt_mask = bert_input['attention_mask']

    # get the label and encode the label
    if self.train:
        labels = self.df.iloc[i]['Labels']
        labels = labels.split()
        labels = [int(l) for l in labels]
        # one-hot-encoding
        labels = one_hot_encoding(self.num_labels, labels)
        return (img, labels, txt_ids, txt_mask, img_id)
    return (img, txt_ids, txt_mask, img_id)