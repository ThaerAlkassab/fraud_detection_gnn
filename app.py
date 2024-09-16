# app.py
from data_preprocessing import load_data, preprocess_data
from gnn_model import create_graph_data, train_model
import webbrowser
import os

def main():
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data('./data/5k.csv')
    df = preprocess_data(df)

    # Step 2: Create graph data for GNN
    print("Creating graph data for GNN...")
    graph_data = create_graph_data(df)

    # Step 3: Train the GNN model
    print("Training GNN model...")
    model = train_model(graph_data)
    print("Model trained successfully!")

    # Step 4: Explain predictions using SHAP
    print("Generating SHAP explanations...")
    from explainability import explain_predictions
    explain_predictions(model, graph_data)

    # Step 5: Open visualization in the browser
    print("Opening visualization...")
    webbrowser.open('http://localhost:8000')

if __name__ == "__main__":
    # Start the web server for visualization
    os.system('python3 -m http.server 8000 &')
    
    # Run the main function
    main()
