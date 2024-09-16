# explainability.py
import shap
import torch
from gnn_model import GNN, create_graph_data
from data_preprocessing import load_data, preprocess_data

df = load_data('../data/5k.csv')
df = preprocess_data(df)
graph_data = create_graph_data(df)

model = GNN(input_dim=graph_data.num_features, hidden_dim=16, output_dim=2)
model.eval()

# Use SHAP to explain predictions
explainer = shap.DeepExplainer(model, graph_data.x)
shap_values = explainer.shap_values(graph_data.x)

# Plot summary of the SHAP values
shap.summary_plot(shap_values, graph_data.x.numpy(), feature_names=['Age', 'Income', 'Account Balance', 'Occupation', 'Risk Tolerance', 'Loan Status'])
