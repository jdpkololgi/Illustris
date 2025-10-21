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
from postprocessing import postprocessing
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

import graphviz
from sklearn.tree import export_graphviz
# Make tensorboard SummaryWriter optional (avoid forcing tensorflow import)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_graph(self, *args, **kwargs):
            pass
        def add_figure(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def flush(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass

class Model():
    def __init__(self, model_type = 'mlp', pplot = False):
        self._net = network()
        self.model_selector(model_type)
        self.pplot = pplot

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
            # Try to write graph to TensorBoard if SummaryWriter is available.
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()
                try:
                    writer.add_graph(self.model, torch.randn(1, 7))
                except Exception:
                    # some models / environments may not support add_graph; ignore
                    pass
                writer.close()
            except Exception:
                # TensorBoard not available or triggers heavy imports (tensorflow) — continue silently
                pass
            
        elif model_type == 'dnn':
            self.model = 'work in progress'

        elif model_type == 'random_forest':
            self.model = Model_classes.Random_Forest()
        else:
            raise ValueError('Model type not recognised')
        
    def run(self, epochs, learning_rate, mode = 'train'):
        '''
        Generic function to run different models
        '''
        # Load the data
        self.pipeline(network_type='Delaunay') # {MST, Complex, Delaunay}

        if isinstance(self.model, Model_classes.MLP):
        
            # Weight each class by the inverse of the frequency it depends on the masscut
            class_weights_tensor = torch.tensor(self.class_weights_prebuff, dtype=torch.float32)#.to(self.model.device)

            # Set the loss and optimiser
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate) #torch.optim.AdamW(self.model.parameters(), lr = learning_rate, weight_decay=0.01) # lr, weight_decay=0.01) # L2 regualarisation to prevent overfitting

            if mode == 'train':    
                # Begin training
                self.model.train_model(criterion=criterion, optimiser=optimiser, train_loader=self.train_loader, val_loader=self.val_loader, epochs=epochs)


                # # Plot the loss
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.plot(self.model.loss_list, 'x-', color='r', label='Training Loss')
                # ax.plot(self.model.validation_loss_list, 'x-', color='b', label='Validation Loss')
                # ax.set_xlabel('Epoch')
                # ax.set_ylabel('Loss')
                # ax.legend()
                # plt.show()

                # # Plot the accuracy
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.plot(self.model.accuracy_list, 'x-', color='r', label='Training Accuracy')
                # ax.plot(self.model.validation_accuracy_list, 'x-', color='b', label='Validation Accuracy')
                # ax.set_xlabel('Epoch')
                # ax.set_ylabel('Accuracy')
                # ax.legend()
                # plt.show()

        elif isinstance(self.model, Model_classes.Random_Forest):
            if mode == 'train':
                # Train the model
                self.model.train_model(train_loader=self.train_loader)
                # Validate the model
                val_accuracy = self.model.validate(val_loader=self.val_loader)
        else:
            raise ValueError('Unsupported model type')

        
        # return self.model.loss_list, self.model.validation_loss_list, self.model.accuracy_list, self.model.validation_accuracy_list

    def test(self):
        '''
        Generic function to test different models
        '''
        if isinstance(self.model, Model_classes.MLP):
            # Begin testing
            all_preds, all_labels, all_prob_preds = self.model.test_model(test_loader=self.test_loader)

        elif isinstance(self.model, Model_classes.Random_Forest):
            # Begin testing
            all_preds, all_labels = self.model.test_model(test_loader=self.test_loader)
        else:
            raise ValueError('Unsupported model type')
        
        # # write table of galaxy indices and their corresponding labels and predictions to csv
        # table = pd.DataFrame({'Galaxy Index': self.test_indices, 'Labels': all_labels, 'Predictions': all_preds})
        # table.to_csv(f'predictions_{self.model.__class__.__name__}.csv', index=False)
        
        ''' # print(all_labels)
            # print(all_preds)
            # print(self.test_loader.dataset.classes)
            # print(self.test_loader.dataset.targets)
            # print(self.test_loader.dataset.features)
            # classes = self.test_loader.dataset.classes
            # targets = self.test_loader.dataset.targets
            # features = self.test_loader.dataset.features
            # dataset = pd.DataFrame(features)
            # dataset = self.data
            # n_features = len(dataset.columns) - 1
            
            # n_rows = n_features//3 + n_features%3
            # n_cols = 3
            # # strip plot for each feature and the 4 classes
            # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
            
            # for i in range(n_features):
            #     sns.stripplot(data=dataset, x='Target', ax=axs[i//3, i%3], y=dataset.columns[i], jitter=0.5, alpha=0.5)
            #     axs[i//3, i%3].set_title(f'{dataset.columns[i]}')
            # plt.tight_layout()
            # plt.plot()'''
        
        # # Violin plot
        # dataset = self.data
        # # # quickly invert the density measures and change labels
        # # dataset['Density'] = 1/dataset['Density']
        # # dataset['KNN Density'] = 1/dataset['KNN Density']
        # # dataset.rename(columns={'Density':'Density Inverse', 'KNN Density':'KNN Density Inverse'}, inplace=True)
        # features = dataset.iloc[:,:-1].values # All columns except the last one
        # # targets = self.data.iloc[:,-1].values # The last column

        # scaler = StandardScaler()
        # scaler = PowerTransformer(method = 'box-cox')
        # dataset.iloc[:,:-1] = scaler.fit_transform(features)
        # print(dataset)

        # n_features = len(dataset.columns) - 1

        # n_rows = (n_features + 2) // 3  # calculate the number of rows needed
        # n_cols = min(n_features, 3)  # use 3 columns or less if n_features < 3

        # # strip plot for each feature and the 4 classes
        # fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))
        # fig1, axs1 = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))

        # # Flatten axs array if there's more than one subplot
        # if n_features > 1:
        #     axs = axs.flatten()
        #     axs1 = axs1.flatten()
        # else:
        #     axs = [axs]
        #     axs1 = [axs1]

        # for i in range(n_features):
        #     sns.violinplot(data=dataset, y=dataset.columns[i], x='Target', ax=axs[i], alpha=0.5) #jitter=0.2,
        #     sns.stripplot(data=dataset, x='Target', ax=axs1[i], y=dataset.columns[i], jitter=0.25, alpha=0.5)
        #     axs[i].set_title(f'{dataset.columns[i]}')
        #     axs1[i].set_title(f'{dataset.columns[i]}')
        #     # axs[i].set_yscale('log')
        #     # upper_95thpercentile = dataset[dataset.columns[i]].quantile(0.95)
        #     # lower_95thpercentile = dataset[dataset.columns[i]].quantile(0.5)
        #     # axs[i].set_ylim(lower_95thpercentile, upper_95thpercentile)
        #     # axs1[i].set_ylim(lower_95thpercentile, upper_95thpercentile)

        # if self.pplot:
        #     # Corner plot for the features against each other color-coded by cosmic web environment (Target)
        #     pairplot = sns.pairplot(dataset, hue='Target', palette='Set1', diag_kind='kde', plot_kws={'alpha':0.5, 's':1}, corner=True)
        #     pairplot.map_lower(sns.kdeplot, levels=4)
        #     # for ax in pairplot.axes.flatten():
        #     #     ax.set_xscale('log')
        #     #     ax.set_yscale('log')


        #     # Remove empty subplots if any
        #     for j in range(n_features, len(axs)):
        #         fig.delaxes(axs[j])
        #         fig1.delaxes(axs1[j])

        #     # axs[n_features-1].set_ylim(-1,1)
        #     # axs1[n_features-1].set_ylim(-1,1)
        #     # axs[n_features-1].set_yscale('log')
        #     # axs1[n_features-1].set_yscale('log')

        #     fig.tight_layout()
        #     fig1.tight_layout()
        #     plt.show()
            
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
        plt.xticks(rotation=45)
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

    def onnx_save(self):
        '''
        Function to visualise the neural network model in ONNX format
        '''
        # Prepare an example input (automatically adjust shape if architecture changes)
        example_input = torch.randn(1, 7) # specific to the Delaunay network

        if isinstance(self.model, Model_classes.MLP):

            # Export the model
            torch.onnx.export(Model_classes.MLP(), example_input, 'model.onnx', verbose=False)

            print(f'Model has been saved to {os.getcwd()}')
        elif isinstance(self.model, Model_classes.Random_Forest):
            # Visualise model using .estimators
            for i in range(1): # arbitary number
                tree = self.model.model.estimators_[i]
                dot_data = export_graphviz(tree,
                                           feature_names=self.data.columns[:-1],
                                           class_names=[str(cls) for cls in self.data['Target'].unique()],
                                           filled=True,
                                           impurity=False,
                                           max_depth=3,
                                           proportion=True)
            graph = graphviz.Source(dot_data)
            graph.render(filename='tree', format='pdf', cleanup=True)


        else:
            raise ValueError('Unsupported model type')

if __name__ == '__main__':
    model = Model(model_type='mlp', pplot=True)
    model.run(epochs=100, learning_rate=1e-5)#1e-5#0.000625#0.00025 # learning rate is not used for random forest
    model.test()

    pp = postprocessing(model.model)
    pp.confusion_matrix()
    pp.precision_recall_f1()
    pp.precision_recall_curve()
    pp.train_valid_loss_accuracy()

    model.cross_correlation()
    model.onnx_save()