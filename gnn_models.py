import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, 15)
        self.gcn2 = GCNConv(15, 15)
        self.gcn3 = GCNConv(15, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = F.relu(self.gcn2(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = self.gcn3(x, edge_index, edge_weight)
        return x

class SimpleGAT(nn.Module):
    '''simple GAT model using the GATLayer
    
    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        num_heads (int): Number of attention heads
    
    '''
    def __init__(self, input_dim, output_dim, num_heads = 1): # keep a default of 1 head
        super(SimpleGAT, self).__init__()

        hidden_dim = 20  # per-head output size
        total_hidden = hidden_dim * num_heads  # total output size if concat=True

        self.gat_layer1 = GATv2Conv(input_dim, hidden_dim, edge_dim=1, heads=num_heads, concat=True)
        self.gat_layer2 = GATv2Conv(total_hidden, hidden_dim, edge_dim=1, heads=num_heads, concat=True)
        self.gat_layer3 = GATv2Conv(total_hidden, hidden_dim, edge_dim=1, heads=num_heads, concat=True)
        self.gat_layer4 = GATv2Conv(hidden_dim * num_heads, output_dim, edge_dim=1, heads=1, concat=False)

        self.gat_dropout = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(total_hidden)
        self.norm2 = nn.LayerNorm(total_hidden)
        self.norm3 = nn.LayerNorm(total_hidden)


    def forward(self, x, edge_index, edge_weight=None, return_attention=False, return_embeddings=False):
        """Forward pass of the GNN model"""

        if return_attention:
            x1, (ei1, attn1) = self.gat_layer1(x, edge_index, edge_attr=edge_weight, return_attention_weights = True)
            x1 = self.gat_dropout(F.relu(self.norm1(x1)))

            x2, (ei2, attn2) = self.gat_layer2(x1, edge_index, edge_attr=edge_weight, return_attention_weights = True)
            x2 = self.gat_dropout(F.relu(self.norm2(x2) + x1))

            x3, (ei3, attn3) = self.gat_layer3(x2, edge_index, edge_attr=edge_weight, return_attention_weights = True)
            x3 = self.gat_dropout(F.relu(self.norm3(x3) + x2))

            y_hat, (ei4, attn4) = self.gat_layer4(x3, edge_index, edge_attr=edge_weight, return_attention_weights = True)
            return y_hat, [(ei1, attn1), (ei2, attn2), (ei3, attn3), (ei4, attn4)]
        
        else:

            x1 = self.gat_dropout(F.relu(self.norm1(self.gat_layer1(x, edge_index, edge_attr=edge_weight))))
            x2 = self.gat_dropout(F.relu(self.norm2(self.gat_layer2(x1, edge_index, edge_attr=edge_weight)) + x1))
            x3 = self.gat_dropout(F.relu(self.norm3(self.gat_layer3(x2, edge_index, edge_attr=edge_weight)) + x2))
            y_hat = self.gat_layer4(x3, edge_index, edge_attr=edge_weight)

            if return_embeddings:
                return y_hat, x3  # return the last hidden layer as well for visualization using UMAP

            return y_hat