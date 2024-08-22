import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict
from tqdm import tqdm


def device_check():
    '''
    Function to check if a GPU is available
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')

    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def plot_confusion_matrix(cm, classes):
    '''
    Plot confusion matrix
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_normalised_confusion_matrix(cm, classes):
    '''
    Plot normalised confusion matrix
    '''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Normalised Confusion Matrix')
    return fig

class MLP(nn.Module):

    def __init__(self, n_features = 7, n_hidden = 17, n_output_classes = 4):
        super().__init__()
        self.device = device_check() # Check if a GPU is available
        # Define the layers using nn.Sequential and OrderedDict for named layers
        self.layer_stack = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_features, n_hidden)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(n_hidden, 15)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(15, 10)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(10, 7)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(7, n_output_classes)),
            ('softmax', nn.Softmax(dim = 1)) # dim=1 to apply softmax along the class dimension
        ]))
        self.to(self.device) # Move the model to the device (GPU or CPU)

    def forward(self, x):
        x = x.to(self.device) # Move the input to the device
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
    
    def train(self, criterion, optimiser, train_loader, val_loader, epochs):
        '''
        Training loop for the model
        '''
        writer = SummaryWriter() # Create a SummaryWriter object to write the loss values to TensorBoard
        self.loss_list = [] # List to store the loss values
        self.validation_loss_list = [] # List to store the validation loss values
        self.layer_stack.train() # Set the model to training mode
        for epoch in range(epochs): # Loop over the epochs. One complete pass through the entire training dataset
            epoch_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar: # tqdm is a progress bar. It shows the progress of the training. Each bar update is a batch
                for features, labels in train_loader: # Loop iterates over batches of data from the train_loader. It provides batches of features and corresponding labels
                    features, labels = features.to(self.device), labels.to(self.device) # Move the data to the device
                    # Forward pass
                    outputs = self.layer_stack(features)
                    loss = criterion(outputs, labels) # Calculate the loss

                    # Backward pass and optimisation
                    optimiser.zero_grad() # Clear the gradients
                    loss.backward() # Compute the gradients
                    optimiser.step() # Update the weights
                    
                    # Update the progress bar
                    pbar.set_postfix({'Loss': loss.item()})
                    epoch_loss += loss.item()
                    pbar.update(1)

            # Calculate the average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.loss_list.append(avg_epoch_loss)
            writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
            self.validate(val_loader) # Validate the model after each epoch
            self.validation_loss_list.append(self.validation_loss)
            writer.add_scalar('Loss/Validation', self.validation_loss, epoch)
        writer.flush()    
        writer.close()

    def test(self, test_loader):
        '''
        Testing loop for the model
        '''
        writer = SummaryWriter() # Create a SummaryWriter object to write values to TensorBoard
        self.layer_stack.eval() # Set the model to evaluation mode, ensuring that dropout and batchnorm layers are not active
        correct = 0 # Counter for the number of correct predictions
        total = 0 # Counter for the total number of predictions 
        all_preds = [] # List to store all the predictions for tensorboard
        all_labels = [] # List to store all the labels for tensorboard

        with torch.no_grad(): # Turn off gradient tracking to speed up the computation and reduce memory usage
            for features, labels in test_loader: # Loop iterates over batches of data from the test_loader. It provides batches of features and corresponding labels
                features, labels = features.to(self.device), labels.to(self.device) # Move the data to the device

                outputs = self.layer_stack(features) # Forward pass
                _, predicted = torch.max(outputs, 1) # Get the class with the highest probability and 1 is the dimension along which to find the maximum
                total += labels.size(0) # Increment the total by the number of labels in the batch
                correct += (predicted == labels).sum().item() # Increment the correct counter by the number of correct predictions in the batch
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        self.test_accuracy = 100 * correct / total # Calculate the accuracy as a percentage
        print(f'Test Accuracy: {self.test_accuracy}%')

        # Compute the confusion matrix
        # print(all_preds)
        # print(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        cm_fig = plot_confusion_matrix(cm, classes=test_loader.dataset.classes) #classes=['Cluster', 'Wall', 'Filament', 'Void'])#
        writer.add_figure('Confusion Matrix/Test', cm_fig, global_step=None)

        # Precision, Recall, F1 Score
        # For each class
        stats = precision_recall_fscore_support(all_labels, all_preds)
        stats_df = pd.DataFrame(list(stats), index=['Precision', 'Recall', 'F1 Score', 'Support'], columns=test_loader.dataset.classes) #columns=['Cluster', 'Wall', 'Filament', 'Void'])
        fig, ax = plt.subplots(figsize=(10, 5))
        print(stats_df.loc['Support'])
        stats_df.drop('Support').T.plot(kind='bar', ax=ax)
        ax.set_title('Precision, Recall and F1 Score')
        ax.set_ylabel('Score')
        ax.legend()
        for p in ax.patches:
            # Bar data to 2 decimal places
            ax.annotate(str(p.get_height().round(2)), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.show()
        writer.add_figure('Precision, Recall and F1 Score', fig, global_step=None)

    def validate(self, val_loader):
        '''
        Validation loop for the model
        '''
        writer = SummaryWriter() # Create a SummaryWriter object to write values to TensorBoard
        self.layer_stack.eval() # Set the model to evaluation mode, ensuring that dropout and batchnorm layers are not active
        correct = 0 # Counter for the number of correct predictions
        total = 0
        validation_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Turn off gradient tracking to speed up the computation and reduce memory usage
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device) # Move the data to the device
                outputs = self.layer_stack(features) # Forward pass
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        self.validation_accuracy = 100 * correct / total # Calculate the accuracy as a percentage
        self.validation_loss = validation_loss / len(val_loader)
        # cm = confusion_matrix(all_labels, all_preds)

        # cm_fig = plot_confusion_matrix(cm)
        # writer.add_figure('Confusion Matrix/Validation', cm_fig) # The global step is the epoch number
        print(f'Validation Accuracy: {self.validation_accuracy}%')
        print(f'Validation Loss: {self.validation_loss}')

    