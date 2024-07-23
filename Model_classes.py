import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import os
import torch
from torch import nn
from collections import OrderedDict


'''
MLP Model Architecture:
+----------------------------+
|          Input             |   (10 features)
+----------------------------+
             |
             v
+----------------------------+
|    Linear Layer (fc1)      |   (Input: 10, Output: 20)
|      Weights: [10x20]      |
+----------------------------+
             |
             v
+----------------------------+
|       ReLU Activation      |
+----------------------------+
             |
             v
+----------------------------+
|    Linear Layer (fc2)      |   (Input: 20, Output: 4)
|      Weights: [20x4]       |
+----------------------------+
             |
             v
+----------------------------+
|      Softmax Activation    |
|     (along dim=1)          |
+----------------------------+
             |
             v
+----------------------------+
|         Output             |   (4 classes)
+----------------------------+
'''

class MLP(nn.Module):
    def __init__(self, n_features = 10, n_hidden = 20, n_output_classes = 4):
        super().__init__()

        # Define the layers using nn.Sequential and OrderedDict for named layers
        self.layer_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_features, n_hidden)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(n_hidden, n_output_classes)),
            ('softmax', nn.Softmax(dim = 1)) # dim=1 to apply softmax along the class dimension
        ]))

    def forward(self, x):
        # Forward propagate input through the layers
        return self.layer_stack(x)