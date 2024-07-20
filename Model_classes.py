import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import os
import torch
from torch import nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, n_features = 10, n_hidden = 20, n_output_classes = 4):
        super().__init__()

        # Define the layers
        self.layer_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_features, n_hidden)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(n_hidden, n_output_classes)),
            ('softmax', nn.Softmax(dim = 1))
        ]))