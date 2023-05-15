import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import *
from torchsummary import summary
from dataset import *

class MultilabelCNN(nn.Module):

  def __init__(self, num_classes, pretrained_model, out_size=None):
    super(MultilabelCNN, self).__init__()
    # define convolutional layers
    self.pretrained_model = pretrained_model
    self.pretrained = nn.Sequential(*(list(self.pretrained_model.children())[:-1])) # remove last layer
    self.classifier = nn.Sequential(nn.Linear(out_size, num_classes),
                                    nn.Sigmoid()
                                    )

  def forward(self, x):
    # pass input through convolutional layers
    out = self.pretrained(x)
    # flatten output from convolutional layers
    out = torch.flatten(out, 1)
    # pass flattened output through fully connected layers
    out = self.classifier(out)
    return out