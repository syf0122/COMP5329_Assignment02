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

# use GPU if possible
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('I am using GPU.' if use_cuda else 'I am using CPU.')

num_labels = 19
batch_size = 32
train_val_split = 0.8
img_size = (224, 224)
transforms = Compose([Resize(img_size),
                      ToTensor()
                      ])

# Load test data
test_dataset = ImageDataset(label_file = 'test.csv',
                            dir = dir,
                            num_labels = num_labels,
                            transform = transforms
                            )
test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=False)