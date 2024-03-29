import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.transforms import *
import torchvision.transforms.functional as functional
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchsummary import summary
from dataset import *
from model import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# use GPU if possible
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('I am using GPU.' if use_cuda else 'I am using CPU.')

# some parameters
num_labels = 19
batch_size = 64
train_val_split = 0.80
img_size = (224, 224)

# preprocessing and augmentation
transforms = Compose([ToTensor(),
                      Resize(img_size, antialias=True),
                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      RandomHorizontalFlip(),
                      RandomVerticalFlip(),
                      RandomRotation(90)
                      ])

# The main dataset available with labels
dataset = ImageDataset(label_file = 'train.csv',
                       dir = 'data/',
                       num_labels = num_labels,
                       transform = transforms
                      )
length = [int(round(train_val_split * len(dataset))), int(round((1 - train_val_split) * len(dataset)))]
train_dataset, val_dataset = random_split(dataset, length) 
# create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# # show one item from the dataset
# sample_img, label, caption = train_dataset[15]
# plt.imshow(sample_img.transpose(0, 1).transpose(1, 2))
# plt.show()
# print(label)
# print(caption)

# some hyperparameters for training
lr = 1e-4
epochs = 5
model_to_use = 'efficientnet_v2_s' # specify the model to use
include_txt = True # whether to include text or not
# initialize the model
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
elif model_to_use == 'efficientnet_v2_s':
    pretrained_model = models.efficientnet_v2_s(pretrained=True)
    model = MultilabelCNN(num_labels, pretrained_model, out_size=1280, multimodal=include_txt)
print(f'Using {model_to_use}, include text: {include_txt}.')

model.to(device)
# summary(model, input_size=(3, 224, 224))
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training and validation
tra_loss_hist = []
tra_f1_scores = []
val_loss_hist = []
val_f1_scores = []

for epo in range(epochs):
    # Training
    model.train()
    epo_loss = []
    progress_bar = tqdm(range(len(train_dataloader)))
    train_pred = []
    train_targ = []
    for i, (img, label, txt_id, txt_mask, id) in enumerate(train_dataloader):
        # send data to device
        img = img.to(device)
        label = label.to(device)
        txt_id = txt_id.squeeze(1).to(device)
        txt_mask = txt_mask.to(device)

        # Zero gradient
        optimizer.zero_grad()
        # Make predictions
        pred = model(img, txt_id, txt_mask)
        # get the labels 
        train_pred.append(one_hot_encoding(num_labels, decode_labels(pred.cpu().tolist()[0], 0.6)))
        train_targ.append(label.cpu().tolist()[0])
        # Compute the loss and its gradients
        loss = loss_func(pred.float(), label.float())
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        epo_loss.append(loss.item())
        progress_bar.update(1)
    progress_bar.close()
    epo_loss = np.array(epo_loss)
    tra_loss_hist.append(epo_loss)

    # Validation
    model.eval()
    epo_loss_val = []
    pred_ls = []
    targ_ls = []
    with torch.no_grad():
        val_loss = 0
        progress_bar = tqdm(range(len(val_dataloader)))
        for img, label, txt_id, txt_mask, id in val_dataloader:
            # send data to device
            img = img.to(device)
            label = label.to(device)
            txt_id = txt_id.squeeze(1).to(device)
            txt_mask = txt_mask.to(device)
            # prediction
            output = model(img, txt_id, txt_mask)
            # calculate loss
            val_loss = loss_func(output.float(), label.float())
            # decode the label 
            pred_ls.append(one_hot_encoding(num_labels, decode_labels(output.cpu().tolist()[0], 0.6)))
            targ_ls.append(label.cpu().tolist()[0])
            epo_loss_val.append(val_loss.item())
            progress_bar.update(1)
        progress_bar.close()
        f1_tra = f1_score(train_pred, train_targ, average='weighted', zero_division=0)
        tra_f1_scores.append(f1_tra)
        f1_val = f1_score(pred_ls, targ_ls, average='weighted', zero_division=0)
        val_f1_scores.append(f1_val)
        epo_loss_val = np.array(epo_loss_val)
        val_loss_hist.append(epo_loss_val)
    # print the loss during training
    print(f'Epoch {epo+1} Train BCELoss = {round(epo_loss.mean(), 3)}, Train F1-score = {round(f1_tra, 3)}, Validation BCELoss = {round(epo_loss_val.mean(), 3)}, Validation F1-score = {round(f1_val, 3)}.\n')

tra_loss_hist = np.array(tra_loss_hist)
val_loss_hist = np.array(val_loss_hist)

if include_txt:
    # save the model and the training/validation history.
    np.save('trained_models/'+model_to_use+'_bert_train_loss.npy', tra_loss_hist)
    np.save('trained_models/'+model_to_use+'_bert_valid_loss.npy', val_loss_hist)
    np.save('trained_models/'+model_to_use+'_bert_train_f1.npy', tra_f1_scores)
    np.save('trained_models/'+model_to_use+'_bert_valid_f1.npy', val_f1_scores)
    torch.save(model.state_dict(), 'trained_models/'+model_to_use+'_bert.pt')
else:
    # save the model and the training/validation history.
    np.save('trained_models/'+model_to_use+'_train_loss.npy', tra_loss_hist)
    np.save('trained_models/'+model_to_use+'_valid_loss.npy', val_loss_hist)
    np.save('trained_models/'+model_to_use+'_train_f1.npy', tra_f1_scores)
    np.save('trained_models/'+model_to_use+'_valid_f1.npy', val_f1_scores)
    torch.save(model.state_dict(), 'trained_models/'+model_to_use+'.pt')