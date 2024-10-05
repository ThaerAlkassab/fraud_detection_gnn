import torch
import matplotlib.pyplot as plt
from src.train import train_with_loss
from src.models.gearsage import GEARSage
from src.data_loader import load_data, prepare_data

def test_loss_curve():
    # Load dataset
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Initialize the model
    model = GEARSage(in_channels=X.shape[1], hidden_channels=64, out_channels=2)

    # Train the model and capture the loss
    losses, _ = train_with_loss(model, data, epochs=10)

    # Plot the loss curve
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title("Loss Curve over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    test_loss_curve()
