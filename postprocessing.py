import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict
from tqdm import tqdm

# Random forest decision tree
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

classes = ['Void (0)', 'Wall (1)', 'Filament (2)', 'Cluster (3)']
numerical_classes = [0, 1, 2, 3]

true_colors = ['red', 'green', 'blue', 'violet']
pred_colors = ['purple', 'orange', 'cyan', 'magenta']

class postprocessing():
    def __init__(self, model):
        self.model = model
        self.training_loss = model.loss_list
        self.training_accuracy = model.accuracy_list
        self.validation_loss = model.validation_loss_list
        self.validation_accuracy = model.validation_accuracy_list
        self.labels = model.all_labels
        self.predicted_labels = model.all_preds
        self.predicted_probs = model.all_prob_preds

        self.coverage_test()

    def coverage_test(self):
        ''' Coverage Test '''
        void_post = self.predicted_probs[:,0]
        wall_post = self.predicted_probs[:,1]
        filament_post = self.predicted_probs[:,2]
        cluster_post = self.predicted_probs[:,3]

        n_bins = 10


        prob_range = (0, 1) # should always be between 0 and 1
        # for each class create probability bins and count the number of samples in each bin but we need to know which bin each sample belongs to. N = number in each probability bin and n = number of galaxies in each bin that are truely that class
        void_hist, void_bins = np.histogram(void_post, bins=n_bins, range=prob_range)
        wall_hist, wall_bins = np.histogram(wall_post, bins=n_bins, range=prob_range)
        filament_hist, filament_bins = np.histogram(filament_post, bins=n_bins, range=prob_range)
        cluster_hist, cluster_bins = np.histogram(cluster_post, bins=n_bins, range=prob_range)

        # how many galaxies are classified as each class in each probability bin
        void_correct = np.histogram(void_post[(self.labels==0) ], bins=void_bins, range=prob_range)[0]
        wall_correct = np.histogram(wall_post[(self.labels==1) ], bins=wall_bins, range=prob_range)[0]
        filament_correct = np.histogram(filament_post[(self.labels==2) ], bins=filament_bins, range=prob_range)[0]
        cluster_correct = np.histogram(cluster_post[(self.labels==3) ], bins=cluster_bins, range=prob_range)[0]

        void_empirical_probs = np.array(void_correct/void_hist)
        wall_empirical_probs = np.array(wall_correct/wall_hist)
        filament_empirical_probs = np.array(filament_correct/filament_hist)
        cluster_empirical_probs = np.array(cluster_correct/cluster_hist)

        # which bin each sample belongs to for each class posterior
        void_bin_indices = np.digitize(void_post, void_bins, right=True)
        wall_bin_indices = np.digitize(wall_post, wall_bins, right=True)
        filament_bin_indices = np.digitize(filament_post, filament_bins, right=True)
        cluster_bin_indices = np.digitize(cluster_post, cluster_bins, right=True)

        #empirical probabilities for each galaxy <n_i>/N from the probability bins
        empirical_probs = np.array([void_empirical_probs[void_bin_indices-1], wall_empirical_probs[wall_bin_indices-1], filament_empirical_probs[filament_bin_indices-1], cluster_empirical_probs[cluster_bin_indices-1]])

        self.A_max = empirical_probs.max(axis=0).mean()
        print(f'Expected Accuracy: {self.A_max*100:.2f}%')

        # assigning error bars by modelling counts as a binomial distribution
        n_bi = np.array([void_hist, wall_hist, filament_hist, cluster_hist])
        k_bi = np.array([void_correct, wall_correct, filament_correct, cluster_correct]) 
        p_bi = np.array([void_empirical_probs, wall_empirical_probs, filament_empirical_probs, cluster_empirical_probs])
        error_bi = np.sqrt(p_bi*(1-p_bi)/n_bi)

        # plot
        fig, ax = plt.subplots(2,2, figsize=(15,10))
        ax[0,0].errorbar(void_bins[:-1], void_correct/void_hist, yerr=error_bi[0], label='Void', color='red', marker='.', linestyle='None')
        ax[0,1].errorbar(wall_bins[:-1], wall_correct/wall_hist, yerr=error_bi[1], label='Wall', color='green', marker='.', linestyle='None')
        ax[1,0].errorbar(filament_bins[:-1], filament_correct/filament_hist, yerr=error_bi[2], label='Filament', color='blue', marker='.', linestyle='None')
        ax[1,1].errorbar(cluster_bins[:-1], cluster_correct/cluster_hist, yerr=error_bi[3], label='Cluster', color='violet', marker='.', linestyle='None')

        ax[0,0].axline([0,0], slope=1, ls='--', color='black')
        ax[0,1].axline([0,0], slope=1, ls='--', color='black')
        ax[1,0].axline([0,0], slope=1, ls='--', color='black')
        ax[1,1].axline([0,0], slope=1, ls='--', color='black')

        ax[0,0].set_ylim(0,1)
        ax[0,1].set_ylim(0,1)
        ax[1,0].set_ylim(0,1)
        ax[1,1].set_ylim(0,1)

        ax[0,0].set_xlim(0,1)
        ax[0,1].set_xlim(0,1)
        ax[1,0].set_xlim(0,1)
        ax[1,1].set_xlim(0,1)

        ax[0,0].legend(loc='upper left')
        ax[0,1].legend(loc='upper left')
        ax[1,0].legend(loc='upper left')
        ax[1,1].legend(loc='upper left')

        ax[0,1].set_xlabel(rf'Predicted Posterior Probability $P(Wall \mid x)$ bins = {n_bins}', fontsize=14)
        ax[0,0].set_xlabel(rf'Predicted Posterior Probability $P(Void \mid x)$ bins = {n_bins}', fontsize=14)
        ax[1,0].set_xlabel(rf'Predicted Posterior Probability $P(Filament \mid x)$ bins = {n_bins}', fontsize=14)
        ax[1,1].set_xlabel(rf'Predicted Posterior Probability $P(Cluster \mid x)$ bins = {n_bins}', fontsize=14)

        # set to latex
        plt.rcParams["text.usetex"] = True

        ax[0,0].set_ylabel(r'Expected $P(k=0 \mid x) = \frac{n_{void}}{N_{bin}}$', fontsize=14)
        ax[0,1].set_ylabel(r'Expected $P(k=1 \mid x) = \frac{n_{wall}}{N_{bin}}$', fontsize=14)
        ax[1,0].set_ylabel(r'Expected $P(k=2 \mid x) = \frac{n_{filament}}{N_{bin}}$', fontsize=14)
        ax[1,1].set_ylabel(r'Expected $P(k=3 \mid x) = \frac{n_{cluster}}{N_{bin}}$', fontsize=14)

    def train_valid_loss_accuracy(self):
        # make sure coverage test has been run first
        assert hasattr(self, 'A_max'), 'Coverage test has not been run, please run this first before plotting'

        # plot
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(self.training_loss , label='Training Loss')
        ax[0].plot(self.validation_loss, label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[1].plot(self.training_accuracy, label='Training Accuracy')
        ax[1].plot(self.validation_accuracy, label='Validation Accuracy')
        ax[1].axhline(self.A_max*100, color='r', linestyle='--', label='Expected Accuracy')
        ax[1].set_ylim([0, 100])
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.show()

    def confusion_matrix(self):
        # normalised confusion matrix
        cm = confusion_matrix(self.labels, self.predicted_labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # plot
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        print(cm)
        return cm
    
    def precision_recall_f1(self):
        stats = classification_report(self.labels, self.predicted_labels, target_names=classes, output_dict=True)
        stats_df = pd.DataFrame(stats).transpose().drop(columns=['support'])

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))
        stats_df.plot(kind='bar', ax=ax)

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() * 1.005, p.get_height() * 1.005))

        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        plt.show()

        print(stats)
        return stats_df
    
    def precision_recall_curve(self):
        y = label_binarize(self.labels, classes=numerical_classes)
        
        precision = dict()
        recall = dict()
        thresholds = dict()

        for i in range(len(classes)):
            precision[i], recall[i], thresholds[i] = precision_recall_curve(y[:, i], self.predicted_probs[:, i])

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(len(classes)):
            ax.plot(recall[i], precision[i], label=classes[i], color=true_colors[i])

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        plt.show()

        return precision, recall, thresholds