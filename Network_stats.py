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
from scipy.spatial import Voronoi, ConvexHull, KDTree
from sklearn.neighbors import KernelDensity

import itertools


from Utilities import cat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.utils import compute_class_weight
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

def volume_tetrahedron(tetrahedron):
    '''
    Function to calculate the volume of a tetrahedron
    '''
    matrix = np.array([
        tetrahedron[0] - tetrahedron[3],
        tetrahedron[1] - tetrahedron[3],
        tetrahedron[2] - tetrahedron[3]
    ])
    return abs(np.linalg.det(matrix) / 6)

vec_volume_tetrahedron = np.vectorize(volume_tetrahedron, signature='(n,m)->()')

def voronoi_density(points):
    """
    Calculate the density based on the Voronoi volume for each point.
    
    Parameters:
    points (np.ndarray): Array of points (N x D).
    
    Returns:
    np.ndarray: Array of densities for each point.
    """
    vor = Voronoi(points)
    densities = np.zeros(points.shape[0])
    
    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 in region:  # Skip regions with infinite vertices
            densities[i] = np.inf
        else:
            vertices = vor.vertices[region]
            if len(vertices) > 0:
                volume = ConvexHull(vertices).volume
                densities[i] = 1 / volume if volume > 0 else np.inf
            else:
                densities[i] = np.inf
    
    return densities

def knn_density(points, k=5):
    """
    Calculate the k-nearest neighbor density for each point.
    
    Parameters:
    points (np.ndarray): Array of points (N x D).
    k (int): Number of nearest neighbors to consider.
    
    Returns:
    np.ndarray: Array of densities for each point.
    """
    tree = KDTree(points)
    densities = np.zeros(points.shape[0])
    
    for i, point in enumerate(points):
        distances, _ = tree.query(point, k=k+1)  # k+1 because the point itself is included
        densities[i] = k / np.sum(distances[1:])  # Exclude the distance to itself
    
    return densities

def kde_density(points, bandwidth=1.0):
    """
    Calculate the density using Kernel Density Estimation (KDE).
    
    Parameters:
    points (np.ndarray): Array of points (N x D).
    bandwidth (float): Bandwidth for KDE.
    
    Returns:
    np.ndarray: Array of densities for each point.
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)
    log_density = kde.score_samples(points)
    return np.exp(log_density)

class network(cat):
    def __init__(self):
        self._utils = cat(path=r'C:\Users\dkter\OneDrive - University College London\Year 1\Illustris\TNG300-1', snapno=99, masscut=1e10)

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
        netx = self.subhalo_complex_network(l=2.92)
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
    
    def network_stats_delaunay(self, weight = 'length', buffer=False):
        '''
        Function to calculate the network statistics for the Delaunay triangulation.
        '''
        netx = self.subhalo_delauany_network(xyzplot=False)
        assert isinstance(netx, nx.Graph), 'Networkx graph not created'

        self.degree = netx.degree(weight=weight)
        # self.average_degree = nx.average_neighbor_degree(netx, weight=weight)
        #self.degree_centrality = nx.degree_centrality(netx)
        # self.betweenness_centrality = nx.betweenness_centrality(netx) #this calculation is too slow
        self.clustering = nx.clustering(netx, weight=weight)

        # Use the edge lengths as features
        self.edge_lengths = {(u, v): netx[u][v]['length'] for u, v in netx.edges()}
        #{(u, v): euclidean(netx.nodes[u]['pos'], netx.nodes[v]['pos']) for u, v in netx.edges()}

        self.neigh_list = [netx.edges(node) for node in netx.nodes()]

        # Average length for each node
        self.mean_elen = [np.mean([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        # self.sum_elen = [np.sum([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        self.min_elen = [np.min([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]
        self.max_elen = [np.max([self.edge_lengths[tuple(sorted(edge))] for edge in self.neigh_list [node]]) for node in range(len(netx.nodes()))]

        # Angle between edges
        #self.angles = dict_angle_between_edges(netx, self.neigh_list) # {node: {(edge1, edge2): angle}}
        #self.mean_angles = {node: np.max(list(self.angles[node].values())) for node in range(len(netx.nodes()))}
        
        # Number of connected triangles
        # self.triangles = nx.triangles(netx)
        
        # Volume of tetrahedra
        # self.tetrahedra = {}
        # for node in netx.nodes():
        #     self.tetrahedra[node] = np.sum([volume_tetrahedron(self.points[self.tri.simplices[np.where(self.tri.simplices == node)[0][0]]]) for node in netx.neighbors(node)])
        node_to_simplices = {node: [] for node in range(len(netx.nodes()))}
        for simplex_index, simplex in enumerate(self.tri.simplices):
            for node in simplex:
                node_to_simplices[node].append(simplex_index)

        simplex_points = {node: self.points[self.tri.simplices[simplices]] for node, simplices in node_to_simplices.items()}
        # self.tetra_dens = {node: 1/(0.25*np.sum(vec_volume_tetrahedron(points))) for node, points in simplex_points.items()} # Density of tetrahedra assuming each node has 1/4 of the volume of the tetrahedra around it
        # self.tetra_dens = {node: len(node_to_simplices[node])/(np.sum(vec_volume_tetrahedron(points))) for node, points in simplex_points.items()} # New density of tetrahedra normalised by the number of tetrahedra around the node. A direct measure of participation in the l
        # self.tetra_dens_degree = {node: self.degree[node]/(np.sum(vec_volume_tetrahedron(points))) for node, points in simplex_points.items()} # 

        self.tetra_dens = {node: (len(simplices) / np.sum(vec_volume_tetrahedron(self.points[self.tri.simplices[simplices]]))) / self.degree[node] for node, simplices in node_to_simplices.items()}# if self.degree[node] > 0} # normalising by degree and number of tetrahedra around the node
        
        
        # Neighbour tetrahedra density
        self.neigh_tetra_dens = {node: np.mean([self.tetra_dens[neigh] for neigh in netx.neighbors(node)]) for node in range(len(netx.nodes()))}
        '''
        
        
        self.tetra_dens_degree = {node: self.degree[node] / np.sum(vec_volume_tetrahedron(points)) for node, points in simplex_points.items()}

        # # Neighbour neighbour tetrahedra density
        # self.neigh_neigh_tetra_dens = {node: np.mean([self.tetra_dens[neigh] for neigh in netx.neighbors(neigh)]) for node, neigh in netx.neighbors(node)}
        '''
        # self.tetra_dens = voronoi_density(self.points)
        # self.tetra_dens_neigh = {node: np.mean([self.tetra_dens[neigh] for neigh in netx.neighbors(node)]) for node in range(len(netx.nodes()))}
        # self.knn_dens = knn_density(self.points, k=5) # k-nearest neighbour density
        #self.data = pd.DataFrame({'Degree': list(dict(self.degree).values()), 'Average Degree': list(self.average_degree.values()), 'Degree Centrality': list(self.degree_centrality.values()), 'Mean E.L.': self.mean_elen, 'Sum E.L.': self.sum_elen, 'Min E.L.': self.min_elen, 'Max E.L.': self.max_elen, 'Clustering': list(self.clustering.values()), 'Max Angle': list(self.mean_angles.values()), 'Triangles': list(self.triangles.values()), 'Target': self.cweb})
        # self.kde_dens = kde_density(self.points)

        # commenting out the inertia eigenvalues for now
        # inertia_eigenvalues = {}
        # for node in netx.nodes:
        #     neighbors = list(netx.neighbors(node))
        #     if len(neighbors) < 3:
        #         inertia_eigenvalues[node] = [0.0, 0.0, 0.0]
        #         continue
        #     nbr_pos = self.points[neighbors]  # shape (N_neighbors, 3)
        #     center = nbr_pos.mean(axis=0)
        #     rel_pos = nbr_pos - center
        #     cov = np.dot(rel_pos.T, rel_pos) / len(neighbors)
        #     eigvals = np.linalg.eigvalsh(cov)  # sorted eigenvalues
        #     inertia_eigenvalues[node] = eigvals.tolist()

        # I_eig1 = [inertia_eigenvalues[i][0] for i in range(len(inertia_eigenvalues))]
        # I_eig2 = [inertia_eigenvalues[i][1] for i in range(len(inertia_eigenvalues))]
        # I_eig3 = [inertia_eigenvalues[i][2] for i in range(len(inertia_eigenvalues))]
        

        # Throw error if self.cweb does not exist
        assert hasattr(self, 'cweb'), 'cweb attribute does not exist, please run the cweb_classify method' 

        # Add in xyz coordinates and use them to remove 10Mpc from each side of the cube before dropping the fields'UB':self.UB, 'BV': self.BV, 'VK':self.VK, 'gr':self.gr, 'ri':self.ri, 'iz':self.iz, 
        # self.data = pd.DataFrame({'Degree': list(dict(self.degree).values()), 'Mean E.L.': self.mean_elen, 'Min E.L.': self.min_elen, 'Max E.L.': self.max_elen, 'Clustering': list(self.clustering.values()), 'Density': np.array(list(self.tetra_dens.values())), 'Neigh Density' : np.array(list(self.neigh_tetra_dens.values())), 'I_eig1': I_eig1, 'I_eig2': I_eig2, 'I_eig3': I_eig3, 'Target': self.cweb}) 
        self.data = pd.DataFrame({'Degree': list(dict(self.degree).values()), 'Mean E.L.': self.mean_elen, 'Min E.L.': self.min_elen, 'Max E.L.': self.max_elen, 'Clustering': list(self.clustering.values()), 'Density': np.array(list(self.tetra_dens.values())), 'Neigh Density' : np.array(list(self.neigh_tetra_dens.values())), 'Target': self.cweb})
        print('length before buffering: ', len(self.data))
        self.data['x'] = self.points[:,0]
        self.data['y'] = self.points[:,1]
        self.data['z'] = self.points[:,2]
        self.class_weights_prebuff = compute_class_weight(class_weight='balanced', classes=np.unique(self.data['Target']), y=self.data['Target'])
        print("Class weights (pre-buffer): ", self.class_weights_prebuff)

        if buffer:
            self.data = self.data[(self.data['x']>10) & (self.data['x']<290) & (self.data['y']>10) & (self.data['y']<290) & (self.data['z']>10) & (self.data['z']<290)]
        self.data = self.data.drop(columns=['x', 'y', 'z'])
        print('length after buffering: ', len(self.data))

        self.data.index.name = 'Node ID'

    def pipeline(self, network_type = 'MST'):
        '''
        Data preprocessing pipeline
        '''
        # Load the data and target
        self.cweb_classify(xyzplot=False)
        if network_type == 'MST':
            self.network_stats()
        elif network_type == 'Complex':
            self.network_stats_complex()
        elif network_type == 'Delaunay':
            self.network_stats_delaunay(buffer=True)
        
        # Creating weights for the classes by the inverse of the frequency
        # self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.data['Target']), y=self.data['Target'])
        # print("Class weights (post buffer): ", self.class_weights)        

        self.data.index.name = 'Node ID'

        # Balancing the dataset by classes
        # class_counts = self.data['Target'].value_counts()
        # min_class = class_counts.idxmin()
        # min_class_count = class_counts.min()
        # self.data = self.data.groupby('Target').sample(min_class_count).sort_index() # Sample the minimum class count randomly from each class

        # Feature scaling
        features = self.data.iloc[:,:-1] # All columns except the last one
        targets = self.data.iloc[:,-1] # The last column

        scaler = StandardScaler()
        scaler = PowerTransformer(method = 'box-cox')
        features = pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

        # Train-test split       
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, stratify=targets)#, random_state=21)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)#, random_state=21)
        # 0.25 x 0.8 = 0.2

        self.train_indices = X_train.index
        self.val_indices = X_val.index
        self.test_indices = X_test.index
        
        # # Balancing only the training set by class
        # train_data = pd.DataFrame(X_train)
        # train_data['Target'] = y_train

        # class_counts = train_data['Target'].value_counts()
        # min_class_count = class_counts.min()

        # train_data = train_data.groupby('Target').sample(min_class_count).sort_index() # Sample the minimum class count randomly from each class

        # X_train = train_data.iloc[:,:-1].values
        # y_train = train_data.iloc[:,-1].values

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        y_val = torch.tensor(y_val.values, dtype=torch.long)
        y_test = torch.tensor(y_test.values, dtype=torch.long)

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
    
        classes = ['Void (0)', 'Wall (1)', 'Filament (2)', 'Cluster (3)']
        #['Knot', 'Filament', 'Wall', 'Void']
        train_dataset = CustomDataset(X_train, y_train, classes) # Create the custom dataset
        val_dataset = CustomDataset(X_val, y_val, classes)
        test_dataset = CustomDataset(X_test, y_test, classes)

        # Create DataLoader objects
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # def save_data(self, path):
    #     '''
    #     Save the data to a file. must be run after one network_stats method execution and before pipeline_from_save
    #     '''
    #     assert hasattr(self, 'data'), 'Data attribute does not exist, please run one of the network_stats methods'

    #     self.data.to_csv(path)

    # def pipeline_from_save(self, network_type = 'MST'):
    #     if network_type == 'MST':
    #         self.data = pd.read_csv('data_mst.csv', index_col='Node ID')
    #     elif network_type == 'Complex':
    #         self.data = pd.read_csv('data_complex.csv', index_col='Node ID')
    #     elif network_type == 'Delaunay':
    #         # load the data
    #         self.data = pd.read_csv('data_delaunay.csv', index_col='Node ID')

    #     # Creating weights for the classes by the inverse of the frequency
    #     self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.data['Target']), y=self.data['Target'])
    #     print("Class weights: ", self.class_weights)        

    #     self.data.index.name = 'Node ID'

    #     # Feature scaling
    #     features = self.data.iloc[:,:-1] # All columns except the last one
    #     targets = self.data.iloc[:,-1] # The last column

    #     scaler = StandardScaler()
    #     scaler = PowerTransformer(method = 'box-cox')
    #     features = pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)

    #     # Train-test split       
    #     X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, stratify=targets)#, random_state=21)
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)#, random_state=21)
    #     # 0.25 x 0.8 = 0.2

    #     self.train_indices = X_train.index
    #     self.val_indices = X_val.index
    #     self.test_indices = X_test.index

    #     # Convert to PyTorch tensors
    #     X_train = torch.tensor(X_train.values, dtype=torch.float32)
    #     X_val = torch.tensor(X_val.values, dtype=torch.float32)
    #     X_test = torch.tensor(X_test.values, dtype=torch.float32)
    #     y_train = torch.tensor(y_train.values, dtype=torch.long)
    #     y_val = torch.tensor(y_val.values, dtype=torch.long)
    #     y_test = torch.tensor(y_test.values, dtype=torch.long)

    #     # Define the classes
    #     # classes = ['3.', '2.', '1.', '0.'] #torch.unique(y_train)

    #     # Create Dataset class
    #     class CustomDataset(Dataset): # Custom dataset class
    #         def __init__(self, features, targets, classes):
    #             self.features = features
    #             self.targets = targets
    #             self.classes = classes

    #         def __len__(self): # Returns the number of samples in the dataset
    #             return len(self.features)
            
    #         def __getitem__(self, idx): # Returns the sample at the given index
    #             return self.features[idx], self.targets[idx]
    #     classes = ['Void (0)', 'Wall (1)', 'Filament (2)', 'Cluster (3)']
    #     #['Knot', 'Filament', 'Wall', 'Void']
    #     train_dataset = CustomDataset(X_train, y_train, classes) # Create the custom dataset
    #     val_dataset = CustomDataset(X_val, y_val, classes)
    #     test_dataset = CustomDataset(X_test, y_test, classes)

    #     # Create DataLoader objects
    #     self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #     self.val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    #     self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

