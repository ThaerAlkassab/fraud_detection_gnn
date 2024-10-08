import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import itertools
import random

def load_data(file_path):
    """
    Loads and preprocesses the credit card fraud dataset.
    :param file_path: Path to the CSV file (creditcard.csv)
    :return: Features (X) and labels (y)
    """
    data = pd.read_csv(file_path)
    
    # Drop rows with missing values
    data = data.dropna()

    # Normalize the 'Amount' feature using StandardScaler
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

    # Drop the 'Time' and original 'Amount' columns as they are not necessary
    data = data.drop(['Time', 'Amount'], axis=1)

    # Prepare features (V1-V28 + normAmount)
    X = data.drop(['Class'], axis=1).values

    # Prepare target (fraud or not)
    y = data['Class'].values

    return X, y

def prepare_data(X, y, edge_type="knn", k=5, threshold=3600):
    """
    Converts features and labels into a PyTorch Geometric Data object with different edge creation strategies.
    :param X: Features
    :param y: Labels
    :param edge_type: Type of graph construction (options: 'knn', 'fully_connected', 'class', 'time', 'normAmount', 'random')
    :param k: Number of nearest neighbors (used for K-NN graph)
    :param threshold: Time difference threshold for the 'time' edge creation strategy
    :return: PyTorch Geometric Data object
    """
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    num_transactions = X.shape[0]
    
    # Create edge_index based on different strategies
    if edge_type == "knn":
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if i != neighbor:
                    edges.append([i, neighbor])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif edge_type == "fully_connected":
        edge_index = torch.tensor(list(itertools.permutations(range(num_transactions), 2)), dtype=torch.long).t().contiguous()

    elif edge_type == "class":
        class_column = y
        edges = []
        for i in range(num_transactions):
            for j in range(i + 1, num_transactions):
                if class_column[i] == class_column[j]:
                    edges.append([i, j])
                    edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif edge_type == "time":
        # Assuming X contains a time column at index 0
        time_column = X[:, 0]
        edges = []
        for i in range(num_transactions):
            for j in range(i + 1, num_transactions):
                time_diff = abs(time_column[i] - time_column[j])
                if time_diff < threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif edge_type == "normAmount":
        norm_amount_column = X[:, -1]
        edges = []
        similarity_threshold = 0.1
        for i in range(num_transactions):
            for j in range(i + 1, num_transactions):
                if abs(norm_amount_column[i] - norm_amount_column[j]) < similarity_threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    elif edge_type == "random":
        num_edges = 100  # Arbitrary number of random edges
        edges = []
        for _ in range(num_edges):
            src = random.randint(0, num_transactions - 1)
            dst = random.randint(0, num_transactions - 1)
            if src != dst:
                edges.append([src, dst])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    else:
        raise ValueError("Unsupported edge creation strategy.")
    
    # Create PyTorch Geometric Data object
    graph_data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    return graph_data
