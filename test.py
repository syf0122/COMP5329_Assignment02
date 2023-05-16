import os
import glob
import re
from io import StringIO
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from dataset import *
from model import *

# use GPU if possible
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('I am using GPU.' if use_cuda else 'I am using CPU.')

num_labels = 19
batch_size = 32
train_val_split = 0.8
img_size = (224, 224)
transforms = Compose([ToTensor(),
                      Resize(img_size, antialias=True),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ])

# Load test data
test_dataset = ImageDataset(label_file = 'test.csv',
                            dir = dir,
                            num_labels = num_labels,
                            transform = transforms
                            )
test_dataloader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

# load the trained model
# initialize the model
model_to_use = 'efficientnet_b4'
include_txt = True
if model_to_use == 'vgg19':
    pretrained_model = models.vgg19(pretrained=True)
    model = MultilabelCNN(num_labels, pretrained_model, out_size=25088, multimodal=include_txt)
elif model_to_use == 'resnet50':
    pretrained_model = models.resnet50(pretrained=True)
    model = MultilabelCNN(num_labels, pretrained_model, out_size=2048, multimodal=include_txt)
elif model_to_use == 'efficientnet_b4':
    pretrained_model = models.efficientnet_b4(pretrained=True)
    model = MultilabelCNN(num_labels, pretrained_model, out_size=1792, multimodal=include_txt)
elif model_to_use == 'alexnet':
    pretrained_model = models.alexnet(pretrained=True)
    model = MultilabelCNN(num_labels, pretrained_model, out_size=9216, multimodal=include_txt)
model.load_state_dict(torch.load('trained_models/'+model_to_use+'_bert.pt'), strict=False)
model.to(device) 

# evaluation
model.eval()
predicted_results = {}
with torch.no_grad():
    progress_bar = tqdm(range(len(test_dataloader)))
    for img, txt_id, txt_mask, id in test_dataloader:
        output = model(img.to(device), txt_id.squeeze(1).to(device), txt_mask.to(device))
        # decode the label 
        predicted_results[id] = one_hot_encoding(num_labels, decode_labels(output.cpu().tolist()[0], 0.5))
        progress_bar.update(1)
    progress_bar.close()

for id in predicted_results:
    print(id)
    print(predicted_results[id])

# write the results to the file
out_filename = 'Predicted_labels.txt'
with open(out_filename, 'w') as f:
    for id in predicted_results:
        f.write(id+',')
        for l in predicted_results:
            f.write(' '+l)
        f.write('\n')