# gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# gnn_model.py
def create_graph_data(df):
    # Use the scaled column names as used in the renaming step
    feature_columns = ['age_scaled', 'income_scaled', 'account_balance_scaled', 'occupation_encoded', 
                       'risk_tolerance_encoded', 'loan_status_encoded', 'loan_term_scaled', 'interest_rate_scaled']

    x = torch.tensor(df[feature_columns].values, dtype=torch.float)
    
    y = torch.tensor(df['fraud'].values, dtype=torch.long)

    # Example edge index
    edge_index = torch.tensor([
        [0, 1], [1, 0],
        [2, 3], [3, 2]
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index.t().contiguous(), y=y)


def train_model(data):
    model = GNN(input_dim=data.num_features, hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch} - Loss: {loss.item()}')

    return model

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    df = load_data('../data/5k.csv')
    df = preprocess_data(df)

    graph_data = create_graph_data(df)
    trained_model = train_model(graph_data)
