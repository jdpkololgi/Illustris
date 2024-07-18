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
        self.eigen_centrality = nx.eigenvector_centrality_numpy(netx)
        # self.betweenness_centrality = nx.betweenness_centrality(netx)
        # self.closeness_centrality = nx.closeness_centrality(netx)
        # self.harmonic_centrality = nx.harmonic_centrality(netx)
        self.clustering = nx.clustering(netx)
        self.square_clustering = nx.square_clustering(netx)
        self.generalized_degree = nx.generalized_degree(netx)
        self.triangles = nx.triangles(netx)