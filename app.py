import argparse
import time
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from src.models.gearsage import GEARSage
from src.data_loader import load_data, prepare_data
from src.train import train_model, train_with_loss
from src.utils import load_model

def plot_fraud_distribution(data):
    """Plot Fraud vs Non-Fraud distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=data)
    plt.title("Fraud vs Non-Fraud Transaction Distribution")
    plt.xlabel("Transaction Class (0 = Non-Fraud, 1 = Fraud)")
    plt.ylabel("Count")
    plt.show()

def plot_correlation_heatmap(data):
    """Plot Correlation Heatmap."""
    plt.figure(figsize=(14, 8))
    corr_matrix = data.drop(['Class'], axis=1).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap of PCA Features (V1 to V28)")
    plt.show()

def plot_time_per_epoch(times):
    """Plot Training Time per Epoch."""
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=range(1, len(times) + 1), y=times)
    plt.title("Time Taken Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.show()

def plot_auc_per_epoch(auc_scores):
    """Plot AUC Score per Epoch."""
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=range(1, len(auc_scores) + 1), y=auc_scores)
    plt.title("AUC Score Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_precision_recall_curve(y_true, y_pred):
    """Plot Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def plot_roc_curve(y_true, y_pred):
    """Plot ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="GNN Fraud Detection Training and Testing")

    parser.add_argument('--action', choices=['train', 'test', 'predict', 'visualize'], required=True,
                        help="Choose the action: 'train', 'test', 'predict', or 'visualize' for plots")
    parser.add_argument('--model', choices=['gearsage', 'tgat'], default='gearsage',
                        help="Choose the model: 'gearsage' or 'tgat'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--file', type=str, default='data/creditcard.csv', help="Path to the dataset")

    args = parser.parse_args()

    # Load dataset
    X, y = load_data(args.file)
    data = prepare_data(X, y)

    # Initialize model
    if args.model == 'gearsage':
        model = GEARSage(in_channels=X.shape[1], hidden_channels=64, out_channels=2)
    else:
        model = TGAT(in_channels=X.shape[1], hidden_channels=64, out_channels=2)

    if args.action == 'train':
        # Train the model and capture time and AUC
        times, auc_scores = train_with_loss(model, data, epochs=args.epochs, lr=args.lr)

        # Plot training time and AUC per epoch
        plot_time_per_epoch(times)
        plot_auc_per_epoch(auc_scores)

    elif args.action == 'test':
        # Test the model
        model = load_model('best_model.pth', GEARSage)  # You can add TGAT as well
        model.eval()
        with torch.no_grad():
            out = model(data)
            y_pred = torch.argmax(out, dim=1).cpu().numpy()
            y_true = data.y.cpu().numpy()

            # Plot evaluation metrics
            plot_confusion_matrix(y_true, y_pred)
            plot_precision_recall_curve(y_true, out.cpu().numpy()[:, 1])
            plot_roc_curve(y_true, out.cpu().numpy()[:, 1])

    elif args.action == 'predict':
        # Make predictions
        model = load_model('best_model.pth', GEARSage)
        model.eval()
        with torch.no_grad():
            predictions = model(data)
            print(predictions)

    elif args.action == 'visualize':
        # Visualize dataset properties
        original_data = pd.read_csv(args.file)
        plot_fraud_distribution(original_data)
        plot_correlation_heatmap(original_data)

if __name__ == "__main__":
    main()
