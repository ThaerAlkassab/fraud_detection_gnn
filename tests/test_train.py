import torch
from src.models.gearsage import GEARSage
from src.data_loader import load_data, prepare_data
from src.train import train_model

def test_training():
    # Load dataset
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Initialize the model
    model = GEARSage(in_channels=X.shape[1], hidden_channels=64, out_channels=2)

    # Train the model for a few epochs
    train_model(model, data, epochs=2)

    # Check if the model has updated its parameters
    for param in model.parameters():
        assert param.grad is not None

if __name__ == "__main__":
    test_training()
