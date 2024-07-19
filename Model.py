import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch

from Network_stats import network

class Model():
    def __init__(self, model_type = 'mlp'):
        self._net = network()
