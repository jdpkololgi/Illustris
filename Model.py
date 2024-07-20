import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import os
import torch
from torch import nn


from Network_stats import network
from Model_classes import MLP

class Model():
    def __init__(self, model_type = 'mlp'):
        self._net = network()
        
        self.model_selector(model_type)
        self.device_check()


    def __getattr__(self, name):
        '''
        Implmenting the __getattr__ method to access the attributes of the Utilities class
        '''
        return getattr(self._net, name)
    
    def device_check(self):
        '''
        Function to check if a GPU is available
        '''
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
    
    def model_selector(self, model_type):
        '''
        Function to select the model type
        '''
        if model_type == 'mlp':
            self.model = self.mlp()
        elif model_type == 'dnn':
            self.model = 'work in progress'
        else:
            raise ValueError('Model type not recognised')
        
    def mlp(self):
        '''
        Function to create the multi-layer perceptron model
        '''
                
        return 0