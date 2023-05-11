import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import *
from torchsummary import summary
from dataset import *

class MultilabelCNN(nn.Module):

  def __init__(self, num_classes):
    super(MultilabelCNN, self).__init__()
    # define convolutional layers
    self.pretrained_model = models.efficientnet_b4(pretrained=True) # pretrained model
    self.pretrained = nn.Sequential(*(list(self.pretrained_model.children())[:-1])) # remove last layer
    
    self.classifier = nn.Sequential(nn.Linear(list(self.pretrained_model.children())[-1][1].in_features, 19),
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