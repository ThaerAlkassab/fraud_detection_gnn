# Fraud Detection Using Graph Neural Networks (GNNs) and Explainable AI (XAI)

This project demonstrates how to use Graph Neural Networks (GNNs) for detecting fraud in synthetic financial data. It integrates **Explainable AI (XAI)** techniques for model interpretability and **D3.js** for interactive graph visualization.

## Project Structure

```bash
fraud_detection_gnn/
│
├── data/
│ └── 5k.csv # Dataset (Synthetic financial data)
├── src/
│ ├── data_preprocessing.py # Data preprocessing and feature engineering
│ ├── gnn_model.py # GNN model definition and training
│ ├── explainability.py # SHAP explainability integration
│ ├── visualization/
│ └── index.html # D3.js visualization template
│ └── graph_visualization.js # Visualization script for GNN results
├── app.py # Main application script to run everything
├── README.md # Project documentation
├── requirements.txt # Required Python libraries
```

## Requirements

To run this project, you will need to install the following dependencies:

1. **Python Packages**: Install the required Python libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **D3.js**: Ensure you have a browser that supports D3.js for visualizing the GNN results.

## Workflow

### Step 1: Data Preprocessing

- The script `data_preprocessing.py` loads the dataset, scales numeric features, and encodes categorical features.
- It outputs preprocessed data, ready for training in the GNN.

### Step 2: Training the GNN

- The script `gnn_model.py` defines the GNN architecture and trains the model on the preprocessed data.
- Relationships between customers (or transactions) are defined as graph edges, and the model is trained to detect potential fraud cases.

### Step 3: Explainability with SHAP

- The script `explainability.py` uses SHAP (SHapley Additive exPlanations) to explain the predictions made by the GNN.
- The SHAP plots show which features are contributing the most to the model's decision.

### Step 4: Visualization

- The script `graph_visualization.js` visualizes the fraud detection results in a graph format using D3.js.
- The `app.py` script automatically starts a local web server and opens the visualization in your browser.

## How to Run

1. **Run the Application**:
   Use the `app.py` script to load the dataset, train the GNN, generate SHAP explanations, and visualize the results. Run:

   ```bash
   python app.py
   ```

2. **View Results**:
   Once the app runs, the visualization will open automatically in your browser.

3. **Explore the Data**:
   The D3.js visualization allows you to interact with the results. Nodes represent customers, and edges represent relationships (e.g., shared transactions or common features). Red nodes indicate fraud cases, while blue nodes indicate legitimate transactions.
