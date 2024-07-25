import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.constants as c
import pandas as pd
import networkx as nx
import seaborn as sns
import scienceplots
from networkx.algorithms import node_classification

from Utilities import cat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch

class network(cat):
    def __init__(self):
        self._utils = cat(path=r'/Users/daksheshkololgi/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 1/Illustris/TNG300-1', snapno=99, masscut=1e10)

    def __getattr__(self, name):
        '''
        Implmenting the __getattr__ method to access the attributes of the Utilities class
        '''
        return getattr(self._utils, name)
    
    
    def network_stats(self):
        '''
        Function to calculate the network statistics, there are an arbitrary number of them
        '''
        netx = self.networkx()
        assert isinstance(netx, nx.Graph), 'Networkx graph not created'
        self.degree = netx.degree()
        self.average_degree = nx.average_neighbor_degree(netx)
        self.katz_centrality = nx.katz_centrality(netx, alpha = 0.02)
        self.degree_centrality = nx.degree_centrality(netx)
        self.eigenvector_centrality = nx.eigenvector_centrality_numpy(netx)
        # self.betweenness_centrality = nx.betweenness_centrality(netx)
        # self.closeness_centrality = nx.closeness_centrality(netx)
        # self.harmonic_centrality = nx.harmonic_centrality(netx)
        # self.clustering = nx.clustering(netx) # Seems to only produce zeros
        # self.square_clustering = nx.square_clustering(netx) # Seems to only produce zeros
        # self.generalized_degree = nx.generalized_degree(netx)
        # self.triangles = nx.triangles(netx)

    def pipeline(self):
        '''
        Data preprocessing pipeline
        '''
        # Load the data and target
        self.cweb(xyzplot=False)
        self.data = pd.DataFrame.from_dict({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Katz Centrality': list(self.katz_centrality.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Eigenvector Centrality': list(self.eigenvector_centrality.values()), 'x': self.posx, 'y': self.posy, 'z': self.posz, 'Target': self.cweb})
        self.data.index.name = 'Node ID'

        # Feature scaling
        features = self.data.iloc[:,:-1].values # All columns except the last one
        targets = self.data.iloc[:,-1].values # The last column

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        # 0.25 x 0.8 = 0.2

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create Dataset class
        class CustomDataset(Dataset): # Custom dataset class
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets

            def __len__(self): # Returns the number of samples in the dataset
                return len(self.features)
            
            def __getitem__(self, idx): # Returns the sample at the given index
                return self.features[idx], self.targets[idx]
            
        train_dataset = CustomDataset(X_train, y_train) # Create the custom dataset
        val_dataset = CustomDataset(X_val, y_val)
        test_dataset = CustomDataset(X_test, y_test)

        # Create DataLoader objects
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)
        self.test_loader = DataLoader(test_dataset, batch_size=32)

