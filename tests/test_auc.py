import torch
from sklearn.metrics import roc_auc_score
from src.data_loader import load_data, prepare_data
from src.utils import load_model
import matplotlib.pyplot as plt

def test_auc():
    # Load data
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Load pre-trained model
    model = load_model('best_model.pth', GEARSage)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        out = model(data)
        auc = roc_auc_score(data.y.cpu().numpy(), out.cpu().numpy()[:, 1])
        print(f"AUC Score: {auc}")

        # Plotting AUC over epochs if available
        plt.plot(range(1, len(out) + 1), out.cpu().numpy()[:, 1])
        plt.title("AUC over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("AUC Score")
        plt.show()

if __name__ == "__main__":
    test_auc()
