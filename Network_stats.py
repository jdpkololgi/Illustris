import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import astropy.units as u
import astropy.constants as c
import pandas as pd
import networkx as nx
import seaborn as sns
import scienceplots

from Utilities import cat

class network(cat):
    def __init__(self):
        self._utils = cat(path=r'/Users/daksheshkololgi/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 1/Illustris/TNG300-1', snapno=99, masscut=1e10)
    
    def __getattr__(self, name):
        return getattr(self._utils, name)
    
    def network_stats(self):
        netx = self.networkx()
        assert isinstance(netx, nx.Graph), 'Networkx graph not created'
        return netx
