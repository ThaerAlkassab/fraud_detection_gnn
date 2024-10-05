import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils import load_model
from src.data_loader import load_data, prepare_data

def test_confusion_matrix():
    # Load data
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Load pre-trained model
    model = load_model('best_model.pth', GEARSage)

    # Evaluate model and plot confusion matrix
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_pred = torch.argmax(out, dim=1).cpu().numpy()
        y_true = data.y.cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

if __name__ == "__main__":
    test_confusion_matrix()
