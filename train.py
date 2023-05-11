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
from torchsummary import summary
from dataset import *
from model import *
import matplotlib.pyplot as plt

# use GPU if possible
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('I am using GPU.' if use_cuda else 'I am using CPU.')

num_labels = 19
batch_size = 32
train_val_split = 0.8
img_size = (224, 224)
transforms = Compose([ToTensor(),
                      Resize(img_size),
                      ])

# The main dataset available with labels
dataset = ImageDataset(label_file = 'train.csv',
                       dir = 'data/',
                       num_labels = num_labels,
                       transform = transforms
                      )
length = [int(round(train_val_split * len(dataset))), int(round((1 - train_val_split) * len(dataset)))]
train_dataset, val_dataset = random_split(dataset, length) 

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

# show one item from the dataset
sample_img, label, caption = train_dataset[15]
plt.imshow(sample_img.transpose(0, 1).transpose(1, 2))
plt.show()
print(label)
print(caption)


# lr = 0.001
# epochs = 20
# # initialize the model
# model = MultilabelCNN(num_labels)
# model.to(device)
# summary(model, input_size=(3, 224, 224))

# loss_func = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# # Training and validation
# loss_hist = []
# for epo in range(epochs):
#     epo_loss = []
#     progress_bar = tqdm(range(len(train_dataloader)))
#     for i, (img, label, cap) in enumerate(train_dataloader):
#         # Zero gradient
#         optimizer.zero_grad()
#         # Make predictions
#         pred = model(img.to(device))
#         # Compute the loss and its gradients
#         loss = loss_func(pred.float(), label.float().to(device))
#         loss.backward()
#         # Adjust learning weights
#         optimizer.step()
#         # Gather data and report
#         epo_loss.append(loss.item())
#         progress_bar.update(1)
#     progress_bar.close()
#     epo_loss = np.array(epo_loss)
#     loss_hist.append(epo_loss)

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         for img, label, cap in val_dataloader:
#             output = model(img)
#             val_loss += loss_func(output, label)
#         val_loss /= len(val_dataloader)
        
#     print(f'Epoch {epo+1} Train BCELoss = {round(epo_loss.mean(), 2)}.\n')
# loss_hist = np.array(loss_hist)
