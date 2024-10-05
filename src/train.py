import torch
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from src.utils import save_model
from src.config import CFG
from src.data_loader import load_data, prepare_data
from src.models.gearsage import GEARSage  # or TGAT

def train_model(model, data, epochs, lr=0.001):
    """
    Trains the GNN model on the dataset.
    :param model: GNN model (GEARSage or TGAT)
    :param data: Preprocessed dataset
    :param epochs: Number of epochs for training
    :param lr: Learning rate
    """
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = loss_fn(out, data.y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Evaluation (compute AUC)
        auc = roc_auc_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy()[:, 1])
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, AUC: {auc}')

        # Save model if it performs well
        if auc > CFG.best_auc:
            save_model(model, 'best_model.pth')
            CFG.best_auc = auc

if __name__ == "__main__":
    # Load and preprocess the dataset
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Initialize the GNN model (you can choose GEARSage or TGAT)
    model = GEARSage(in_channels=X.shape[1], hidden_channels=64, out_channels=2)

    # Train the model
    train_model(model, data, epochs=50)
