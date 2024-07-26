import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import os
import torch
from torch import nn

from Network_stats import network
import Model_classes
class Model():
    def __init__(self, model_type = 'mlp'):
        self._net = network()
        self.model_selector(model_type)

    def __getattr__(self, name):
        '''
        Implmenting the __getattr__ method to access the attributes of the Utilities class
        '''
        return getattr(self._net, name)
    
    def model_selector(self, model_type):
        '''
        Function to select the model type
        '''
        if model_type == 'mlp':
            self.model = Model_classes.MLP()
        elif model_type == 'dnn':
            self.model = 'work in progress'
        else:
            raise ValueError('Model type not recognised')
        
    def run(self, epochs, learning_rate, mode = 'train'):
        '''
        Generic function to run different models
        '''
        # Set the loss and optimiser
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr = learning_rate)

        # Load the data
        self.pipeline()

        if mode == 'train':    
            # Begin training
            self.model.train(criterion=criterion, optimiser=optimiser, train_loader=self.train_loader, val_loader=self.val_loader, epochs=epochs)
            # Plot the loss
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.model.loss_list, 'x-', color='r', label='Training Loss')
            ax.plot(self.model.validation_loss_list, 'x-', color='b', label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.show()

    def test(self):
            # Begin testing
            self.model.test(test_loader=self.test_loader)

if __name__ == '__main__':
    model = Model(model_type='mlp')
    model.run(epochs=10, learning_rate=0.0001)
    model.test()