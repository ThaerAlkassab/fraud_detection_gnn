import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GEARSage(torch.nn.Module):
    """
    GEARSage Model: A GraphSAGE-based architecture for handling large-scale dynamic graphs
    such as credit card fraud detection.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GEARSage, self).__init__()
        # Define the first GraphSAGE convolution layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # Define the second GraphSAGE convolution layer
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply first convolution and ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Apply second convolution and softmax for classification
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
