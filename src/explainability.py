# explainability.py

import shap
import torch
from gnn_model import GNN
from data_preprocessing import load_data, preprocess_data

# Load and preprocess data
df = load_data('./data/5k.csv')
df = preprocess_data(df)

# Check for non-numeric columns
print(df.dtypes)  # This will print the data types of each column to ensure they are numeric

# Assume the model is already trained
model = GNN(input_dim=df.shape[1], hidden_dim=16, output_dim=2)
model.eval()

# Convert DataFrame to tensor
graph_data = torch.tensor(df.values, dtype=torch.float)

# Use SHAP to explain predictions
explainer = shap.DeepExplainer(model, graph_data)
shap_values = explainer.shap_values(graph_data)

# Plot summary of the SHAP values
shap.summary_plot(shap_values, graph_data.numpy(), feature_names=df.columns)
