import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import *
from torchsummary import summary
from dataset import *
from transformers import BertModel
# BERT learned and adapted from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

class MultilabelCNN(nn.Module):

  def __init__(self, num_classes, pretrained_model, out_size=None, dropout=0.5, multimodal=True):
    '''
        num_classes: number of classes for the classification
        pretrained_model: loaded pretrained model 
        out_size: output size of the pretrained model, which is the input size to the FC layer
        dropout: dropout rate
        multimodal: whether to include text or not
    '''
    super(MultilabelCNN, self).__init__()
    self.multimodal = multimodal
    # define convolutional layers for image feature extraction
    self.pretrained_model = pretrained_model
    self.pretrained_model = nn.Sequential(*(list(self.pretrained_model.children())[:-1])) # remove last layer
    self.dropout = nn.Dropout(dropout)
    if multimodal:
      # Transformer for text analysis (Bert used)
      self.textmodel = BertModel.from_pretrained('bert-base-cased')

      # classifier
      self.classifier = nn.Sequential(nn.BatchNorm1d(out_size+768),
                                      nn.Linear(out_size+768, num_classes),
                                      nn.Sigmoid()
                                      )
    else:
      # img classifier
      self.img_classifier = nn.Sequential(nn.BatchNorm1d(out_size),
                                          nn.Linear(out_size, num_classes),
                                          nn.Sigmoid()
                                          )

  def forward(self, x, txt_id, txt_mask):
    # image features
    img_feat = self.pretrained_model(x)
    img_feat = self.dropout(img_feat)
    # flatten features
    img_feat = torch.flatten(img_feat, 1)

    if self.multimodal:
      # text features 
      _, txt_feat = self.textmodel(input_ids= txt_id, attention_mask=txt_mask,return_dict=False)
      txt_feat = self.dropout(txt_feat)

      # get all features
      all_feat =  torch.cat([img_feat, txt_feat], dim=1)
      
      # pass flattened output through fully connected layers
      out = self.classifier(all_feat)
    else:
      # img only
      out = self.img_classifier(img_feat)
    return out
