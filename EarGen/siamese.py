# siamese.py

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import random
import numpy as np
import os
from PIL import Image

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # Input is 100x100x3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1),  # padding is added to match 'same' padding in Keras
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1),  # padding is added to match 'same' padding in Keras
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # padding is added to match 'same' padding in Keras
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256*6*6, 4096),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.embedding(x)
        return x

    def forward_two(self, input1, input2):
        l1_distance = torch.abs(input1 - input2)
        output = self.classifier(l1_distance)
        return output


    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        l1_distance = torch.abs(output1 - output2)
        output = self.classifier(l1_distance)
        return output

