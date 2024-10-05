import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from src.utils import load_model
from src.data_loader import load_data, prepare_data

def test_roc_curve():
    # Load data
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Load pre-trained model
    model = load_model('best_model.pth', GEARSage)

    # Evaluate model and plot ROC curve
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = data.y.cpu().numpy()
        y_pred = out.cpu().numpy()[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, marker='.')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

if __name__ == "__main__":
    test_roc_curve()
