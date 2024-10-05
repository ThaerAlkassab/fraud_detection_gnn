import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch

def load_data(file_path):
    """
    Loads and preprocesses the credit card fraud dataset.
    :param file_path: Path to the CSV file (creditcard.csv)
    :return: Features (X) and labels (y)
    """
    data = pd.read_csv(file_path)

    # Normalize the 'Amount' feature using StandardScaler
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

    # Drop the 'Time' and original 'Amount' columns as they are not necessary
    data = data.drop(['Time', 'Amount'], axis=1)

    # Prepare features (V1-V28 + normAmount)
    X = data.drop(['Class'], axis=1).values

    # Prepare target (fraud or not)
    y = data['Class'].values

    return X, y

def prepare_data(X, y):
    """
    Converts features and labels into a PyTorch Geometric Data object.
    :param X: Features
    :param y: Labels
    :return: PyTorch Geometric Data object
    """
    # Convert features and labels to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create a Data object for GNN processing
    data = Data(x=X_tensor, y=y_tensor)
    return data
