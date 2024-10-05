import torch
from sklearn.metrics import roc_auc_score
from src.utils import load_model
from src.data_loader import load_data, prepare_data

def test_model(model, data):
    """
    Evaluates the GNN model on the test dataset.
    :param model: Pre-trained GNN model
    :param data: Preprocessed test dataset
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        out = model(data)
        auc = roc_auc_score(data.y.cpu().detach().numpy(), out.cpu().detach().numpy()[:, 1])
        print(f'Test AUC: {auc}')

if __name__ == "__main__":
    # Load and preprocess the dataset
    X, y = load_data('data/creditcard.csv')
    data = prepare_data(X, y)

    # Load the pre-trained model
    model = load_model('best_model.pth', GEARSage)

    # Test the model
    test_model(model, data)
