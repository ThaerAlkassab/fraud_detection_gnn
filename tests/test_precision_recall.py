import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.utils import load_model
from src.data_loader import load_data, prepare_data

def test_precision_recall_curve():
    # Load data
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Load pre-trained model
    model = load_model('best_model.pth', GEARSage)

    # Evaluate model and plot precision-recall curve
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = data.y.cpu().numpy()
        y_pred = out.cpu().numpy()[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred)

        plt.plot(recall, precision, marker='.')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.show()

if __name__ == "__main__":
    test_precision_recall_curve()
