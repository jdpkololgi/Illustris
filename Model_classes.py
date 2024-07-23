import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import os
import torch
from torch import nn
from collections import OrderedDict
from tqdm import tqdm




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
    
    def __repr__(self, draw = False):
        '''This method is called when the repr() method is called on the object'''
        arc = 'MLP Model Architecture:\n' + self.layer_stack.__repr__()
        ptpic = '''
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
        
        if draw:
            return arc + ptpic
        else:
            return arc

    def __str__(self):
        '''This method is called when the print() method is called on the object'''
        return self.__repr__()
    
    def train(self, criterion, optimiser, train_loader, epochs):
        '''
        Training loop for the model
        '''
        self.layer_stack.train() # Set the model to training mode
        for epoch in range(epochs): # Loop over the epochs. One complete pass through the entire training dataset
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar: # tqdm is a progress bar. It shows the progress of the training. Each bar update is a batch
                for features, labels in train_loader: # Loop iterates over batches of data from the train_loader. It provides batches of features and corresponding labels
                    # Forward pass
                    outputs = self.layer_stack(features)
                    loss = criterion(outputs, labels) # Calculate the loss

                    # Backward pass and optimisation
                    optimiser.zero_grad() # Clear the gradients
                    loss.backward() # Compute the gradients
                    optimiser.step() # Update the weights
                    
                    # Update the progress bar
                    pbar.set_postfix({'Loss': loss.item()})
                    pbar.update(1)

    def test(self, test_loader):
        '''
        Testing loop for the model
        '''
        self.layer_stack.eval() # Set the model to evaluation mode, ensuring that dropout and batchnorm layers are not active
        correct = 0 # Counter for the number of correct predictions
        total = 0 # Counter for the total number of predictions 

        with torch.no_grad(): # Turn off gradient tracking to speed up the computation and reduce memory usage
            for features, labels in test_loader: # Loop iterates over batches of data from the test_loader. It provides batches of features and corresponding labels
                outputs = self.layer_stack(features) # Forward pass
                _, predicted = torch.max(outputs, 1) # Get the class with the highest probability and 1 is the dimension along which to find the maximum
                total += labels.size(0) # Increment the total by the number of labels in the batch
                correct += (predicted == labels).sum().item() # Increment the correct counter by the number of correct predictions in the batch

        self.test_accuracy = 100 * correct / total # Calculate the accuracy as a percentage
        print(f'Test Accuracy: {self.test_accuracy}%')


    def validate(self, val_loader):
        '''
        Validation loop for the model
        '''