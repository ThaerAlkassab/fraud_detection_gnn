import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TGAT(torch.nn.Module):
    """
    TGAT Model: Temporal Graph Attention Network for dynamic graph-based fraud detection.
    It introduces temporal encoding and attention mechanisms.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TGAT, self).__init__()
        # Define the first GAT layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        # Define the second GAT layer
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply first GAT layer with attention mechanism
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Apply second GAT layer and softmax
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
