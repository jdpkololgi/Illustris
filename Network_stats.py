import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.constants as c
import pandas as pd
import networkx as nx
import seaborn as sns
import scienceplots
from scipy.spatial.distance import euclidean, minkowski
import itertools


from Utilities import cat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch

def angle_between_edges(edge1, edge2, netx):
    '''
    Function to calculate the angle between two edges
    '''
    # Get the nodes of the edges
    u1, v1 = edge1
    u2, v2 = edge2

    # Get the positions of the nodes
    pos1 = netx.nodes[u1]['pos'], netx.nodes[v1]['pos']
    pos2 = netx.nodes[u2]['pos'], netx.nodes[v2]['pos']

    # Get the vectors of the edges
    vec1 = pos1[1] - pos1[0]
    vec2 = pos2[1] - pos2[0]

    # Calculate the angle between the edges
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    return angle

def dict_angle_between_edges(netx, neigh_list):
    empty = {}
    for node in netx.nodes():
        empty[node] = {}
        for edge_duo in itertools.combinations(neigh_list[node], 2):
            empty[node][edge_duo] = angle_between_edges(edge_duo[0], edge_duo[1], netx)

    return empty

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
        self.constraint = nx.constraint(netx)

        # Incorporating edge lengths into the feature set
        # Average length for each node
        self.mean_elen = np.zeros(len(self.d))
        for i in range(len(self.l)):
            ind1, ind2 = self.l_index[0][i], self.l_index[1][i]
            self.mean_elen[ind1] += self.l[i]
            self.mean_elen[ind2] += self.l[i]

        self.mean_elen /= self.d

        # Incorporating node neighbour edge lengths into the feature set
        # Average edge length for the neighbours of each node
        self.mean_neigh_elen = np.zeros(len(self.d))
        for i in range(len(self.d)):
            neigh = list(netx.neighbors(i))
            # neigh.append(i) # Include the node itself
            elen_neigh = list(map(self.mean_elen.__getitem__, neigh))
            self.mean_neigh_elen[i] = np.mean(elen_neigh)

        # Incorporating node neighbour's neighbour edge lengths into the feature set
        # Average edge length for the neighbour's neighbours of each node
        self.mean_neigh_neigh_elen = np.zeros(len(self.d))
        for i in range(len(self.d)):
            neigh = list(netx.neighbors(i))
            neigh_neigh = [list(netx.neighbors(j)) for j in neigh] # Neighbours of neighbours
            neigh_neigh = [item for sublist in neigh_neigh for item in sublist] # Flatten the list
            # Commenting next line out so we include the node and its neighbours in the calculation
            neigh_neigh = list(set(neigh_neigh) - set(neigh)) # Remove the neighbours
            elen_neigh_neigh = list(map(self.mean_elen.__getitem__, neigh_neigh)) # Get the edge lengths
            self.mean_neigh_neigh_elen[i] = np.mean(elen_neigh_neigh)

        # # Incorporating node neighbour's neighbour's neighbour edge lengths into the feature set
        # # Average edge length for the neighbour's neighbour's neighbours of each node
        # self.mean_neigh_neigh_neigh_elen = np.zeros(len(self.d))
        # for i in range(len(self.d)):
        #     neigh = list(netx.neighbors(i))
        #     neigh_neigh = [list(netx.neighbors(j)) for j in neigh]
        #     neigh_neigh = [item for sublist in neigh_neigh for item in sublist]
        #     neigh_neigh = list(set(neigh_neigh) - set(neigh))
        #     neigh_neigh_neigh = [list(netx.neighbors(j)) for j in neigh_neigh] # Neighbours of neighbours of neighbours
        #     neigh_neigh_neigh = [item for sublist in neigh_neigh_neigh for item in sublist] # Flatten the list
        #     neigh_neigh_neigh = list(set(neigh_neigh_neigh) - set(neigh_neigh)) # Remove the neighbours
        #     elen_neigh_neigh_neigh = list(map(self.mean_elen.__getitem__, neigh_neigh_neigh)) # Get the edge lengths
        #     self.mean_neigh_neigh_neigh_elen[i] = np.mean(elen_neigh_neigh_neigh)
        self.data = pd.DataFrame.from_dict({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Katz Centrality': list(self.katz_centrality.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Eigenvector Centrality': list(self.eigenvector_centrality.values()), 'Mean Edge Length': self.mean_elen, 'Mean Neighbour Edge Length': self.mean_neigh_elen, 'Mean 2nd Degree Neighbour Edge Length': self.mean_neigh_neigh_elen, 'Target': self.cweb})
        self.data.index.name = 'Node ID'

    def network_stats_complex(self):
        '''
        Function to calculate the network statistics, there are an arbitrary number of them
        '''
        netx = self.subhalo_complex_network()
        assert isinstance(netx, nx.Graph), 'Networkx graph not created'
        self.degree = netx.degree()
        self.average_degree = nx.average_neighbor_degree(netx)
        # self.katz_centrality = nx.katz_centrality(netx, alpha = 0.02)
        self.degree_centrality = nx.degree_centrality(netx)
        self.eigenvector_centrality = nx.eigenvector_centrality_numpy(netx)
        # self.betweenness_centrality = nx.betweenness_centrality(netx)
        # self.closeness_centrality = nx.closeness_centrality(netx)
        # self.harmonic_centrality = nx.harmonic_centrality(netx)
        self.clustering = nx.clustering(netx)
        
        self.data = pd.DataFrame.from_dict({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Eigenvector Centrality': list(self.eigenvector_centrality.values()), 'Clustering': list(self.clustering.values()), 'Target': self.cweb})
        self.data.index.name = 'Node ID'
    
    def network_stats_delaunay(self, weight = 'length'):
        '''
        Function to calculate the network statistics for the Delaunay triangulation.
        '''
        netx = self.subhalo_delauany_network(xyzplot=False)
        assert isinstance(netx, nx.Graph), 'Networkx graph not created'

        self.degree = netx.degree(weight=weight)
        self.average_degree = nx.average_neighbor_degree(netx, weight=weight)
        self.degree_centrality = nx.degree_centrality(netx)
        # self.betweenness_centrality = nx.betweenness_centrality(netx) #this calculation is too slow
        self.clustering = nx.clustering(netx, weight=weight)

        # Use the edge lengths as features
        self.edge_lengths = {(u, v): netx[u][v]['length'] for u, v in netx.edges()}
        #{(u, v): euclidean(netx.nodes[u]['pos'], netx.nodes[v]['pos']) for u, v in netx.edges()}

        self.neigh_list = [netx.edges(node) for node in netx.nodes()]

        # Average length for each node
        self.mean_elen = [np.mean([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        self.sum_elen = [np.sum([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        self.min_elen = [np.min([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        self.max_elen = [np.max([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]

        # Angle between edges
        self.angles = dict_angle_between_edges(netx, self.neigh_list) # {node: {(edge1, edge2): angle}}
        self.mean_angles = {node: np.max(list(self.angles[node].values())) for node in range(len(netx.nodes()))}
        
        # Solid angle for each tetrahedron
        

        # for node in range(len(netx.nodes())):
        #     for edge1 in self.neigh_list[node]:
        #         for edge2 in self.neigh_list[node]:
        #             if edge1 != edge2:
        #                 angle = angle_between_edges(edge1, edge2, netx)
        #                 self.angles[node] = angle
        


        self.data = pd.DataFrame({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Mean E.L.': self.mean_elen, 'Sum E.L.': self.sum_elen, 'Min E.L.': self.min_elen, 'Max E.L.': self.max_elen, 'Clustering': list(self.clustering.values()), 'Max Angle': list(self.mean_angles.values()), 'Target': self.cweb})
        self.data.index.name = 'Node ID'

    def pipeline(self, network_type = 'MST'):
        '''
        Data preprocessing pipeline
        '''
        # Load the data and target
        self.cweb(xyzplot=False)
        if network_type == 'MST':
            self.network_stats()
        elif network_type == 'Complex':
            self.network_stats_complex()
        elif network_type == 'Delaunay':
            self.network_stats_delaunay()
        # self.data = pd.DataFrame.from_dict({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Katz Centrality': list(self.katz_centrality.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Eigenvector Centrality': list(self.eigenvector_centrality.values()), 'x': self.posx, 'y': self.posy, 'z': self.posz, 'Target': self.cweb})
        # self.data = pd.DataFrame.from_dict({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Katz Centrality': list(self.katz_centrality.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Eigenvector Centrality': list(self.eigenvector_centrality.values()), 'Mean Edge Length': self.mean_elen, 'Mean Neighbour Edge Length': self.mean_neigh_elen, 'Mean 2nd Degree Neighbour Edge Length': self.mean_neigh_neigh_elen, 'Target': self.cweb})

        self.data.index.name = 'Node ID'

        # Balancing the dataset by classes
        # class_counts = self.data['Target'].value_counts()
        # min_class = class_counts.idxmin()
        # min_class_count = class_counts.min()
        # self.data = self.data.groupby('Target').sample(min_class_count).sort_index() # Sample the minimum class count randomly from each class

        # Feature scaling
        features = self.data.iloc[:,:-1].values # All columns except the last one
        targets = self.data.iloc[:,-1].values # The last column

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        # 0.25 x 0.8 = 0.2

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Define the classes
        # classes = ['3.', '2.', '1.', '0.'] #torch.unique(y_train)

        # Create Dataset class
        class CustomDataset(Dataset): # Custom dataset class
            def __init__(self, features, targets, classes):
                self.features = features
                self.targets = targets
                self.classes = classes

            def __len__(self): # Returns the number of samples in the dataset
                return len(self.features)
            
            def __getitem__(self, idx): # Returns the sample at the given index
                return self.features[idx], self.targets[idx]
        classes = ['3.', '2.', '1.', '0.']
        #['Knot', 'Filament', 'Wall', 'Void']
        train_dataset = CustomDataset(X_train, y_train, classes) # Create the custom dataset
        val_dataset = CustomDataset(X_val, y_val, classes)
        test_dataset = CustomDataset(X_test, y_test, classes)

        # Create DataLoader objects
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)    