import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

import os
import torch
from torch import nn

from Network_stats import network
import Model_classes
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

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
        self.pipeline(network_type='MST')

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

    def cross_correlation(self):
        '''
        Function to calculate the cross-correlation of the features
        '''
        writer = SummaryWriter()
        features = self.data.drop('Target', axis=1)
        corr = features.corr()

        # Pearson for cross-correlation of the features
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap='Blues', ax=ax, xticklabels=corr.columns, yticklabels=corr.columns)
        ax.set_title('Pearson Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()
        writer.add_figure('Cross Correlation Matrix', fig)

        # Spearman for correlation of the features with the target
        corr_target = features.apply(lambda x: np.abs(x.corr(self.data['Target'], method='spearman')))
        fig, ax = plt.subplots(figsize=(10, 5))
        corr_target.plot(kind='bar', ax=ax)
        ax.set_title('Spearman Correlation')
        ax.set_ylabel('Correlation')
        for p in ax.patches:
            # Bar data to 2 decimal places
            ax.annotate(str(p.get_height().round(2)), (p.get_x() * 1.005, p.get_height() * 1.005))
        plt.show()
        writer.add_figure('Spearman Correlation', fig)

        # # Precision, Recall and F1 Score for each class
        # stats = self.model.precision_recall_f1_score(test_loader=self.test_loader)
        # fig, ax = plt.subplots(figsize=(10, 5))
        # stats.plot(kind='bar', ax=ax)
        # ax.set_title('Precision, Recall and F1 Score')
        # ax.set_ylabel('Score')
        # for p in ax.patches:
        #     # Bar data to 2 decimal places
        #     ax.annotate(str(p.get_height().round(2)), (p.get_x() * 1.005, p.get_height() * 1.005))
        # plt.show()
        # writer.add_figure('Precision, Recall and F1 Score', fig)




if __name__ == '__main__':
    model = Model(model_type='mlp')
    model.run(epochs=25, learning_rate=0.00025)
    model.test()
    model.cross_correlation()