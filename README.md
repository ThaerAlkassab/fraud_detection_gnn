# Credit Card Fraud Detection using Graph Neural Networks (GNN) and Explainable AI (XAI)

This project demonstrates how to use **Graph Neural Networks (GNNs)** to detect fraudulent credit card transactions, leveraging **Explainable AI (XAI)** techniques such as **SHAP** and **GNNExplainer** for better interpretability. The models used include **GEARSage** and **TGAT**, and the project provides comprehensive visualizations for data exploration and model performance.

## Project Overview

- **Data Preprocessing**: Loading and preprocessing the credit card fraud dataset.
- **Models**: Implementations of GNNs, including **GEARSage** and **TGAT**.
- **Training and Testing**: Scripts to train and test the models.
- **Visualizations**: Visualizing performance metrics such as AUC, loss curves, confusion matrices, precision-recall curves, and ROC curves.
- **Unit Tests**: A suite of test files to validate and visualize various metrics.

### Key Features:

- **Explainability**: Integration of SHAP and GNNExplainer to explain model decisions.
- **Flexible Training**: Train different GNN models with customizable parameters.
- **Evaluation Metrics**: Evaluate models using AUC, loss, precision-recall, and confusion matrix.
- **Visualization**: Seaborn and Matplotlib are used to create insightful visualizations.

## Project Structure

```bash
fraud-detection-gnn/
│
├── data/                       # Dataset folder
│   └── creditcard.csv          # Credit card transaction dataset
│
├── src/                        # Source code for models, data loading, and training
│   ├── models/                 # Model implementations
│   │   ├── gearsage.py         # GEARSage model
│   │   └── tgat.py             # TGAT model
│   ├── data_loader.py          # Code to load and preprocess the dataset
│   ├── train.py                # Training script for the models
│   ├── test.py                 # Testing and evaluation script
│   ├── utils.py                # Helper functions (saving/loading models)
│   └── config.py               # Configuration for hyperparameters and settings
│
├── notebooks/                  # Jupyter Notebooks for experimentation
│   └── fraud_detection.ipynb   # Main notebook for GNN fraud detection
│
├── tests/                      # Unit tests and evaluation scripts
│   ├── test_train.py           # Test the training process
│   ├── test_auc.py             # Test and visualize AUC scores
│   ├── test_loss.py            # Test and plot loss curve
│   ├── test_confusion_matrix.py# Plot and test confusion matrix
│   ├── test_precision_recall.py# Test and plot precision-recall curve
│   └── test_roc.py             # Test and plot ROC curve
│
├── img/                        # Images used in the project and thesis
│   └── university_logo.png     # University logo
│
├── app.py                      # Main application entry point for training and testing
├── LICENSE                     # License file (Apache 2.0)
├── README.md                   # Project documentation
└── requirements.txt            # List of required packages for the project
```

## Dataset

The dataset used in this project contains anonymized credit card transactions and can be downloaded from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

**Important Note**: The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

## Installation

Follow these steps to set up the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ThaerAlkassab/fraud_detection_gnn.git
   cd fraud-detection-gnn
   ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place the dataset** in the `data/` directory:
   - The dataset file should be named `creditcard.csv`.

## Usage

You can interact with this project using the provided `app.py` script, which allows you to train, test, and predict using GNN models, along with visualizing dataset characteristics and model performance.

### 1. **Training the GNN Model**

To train the model, use the `app.py` script with the `--action train` flag:

```bash
python app.py --action train --model gearsage --epochs 50 --lr 0.001
```

This command will train the **GEARSage** model for 50 epochs with a learning rate of 0.001. You can modify the parameters or choose the **TGAT** model by changing the `--model` flag.

### 2. **Testing the Model**

Once the model is trained, you can test it using the `app.py` script with the `--action test` flag:

```bash
python app.py --action test --model gearsage
```

This command will evaluate the **GEARSage** model on the test dataset and generate visualizations for the following:

- **Confusion Matrix**
- **Precision-Recall Curve**
- **ROC Curve**

### 3. **Making Predictions**

To make predictions on new data, you can use the `app.py` script with the `--action predict` flag:

```bash
python app.py --action predict --model gearsage
```

### 4. **Visualizing Dataset Characteristics**

To explore the dataset's characteristics such as the **Fraud vs Non-Fraud distribution** and **Correlation Heatmap**, you can run the `--action visualize` command:

```bash
python app.py --action visualize --file data/creditcard.csv
```

This command will generate visualizations for the following:

- **Fraud vs Non-Fraud Distribution**
- **Correlation Heatmap**

### 5. **Visualizing Model Performance**

When you train the model, the script also generates visualizations of the model’s performance during training:

- **Training Time per Epoch**
- **AUC Score per Epoch**

These are displayed automatically after training.

### 6. **Running Unit Tests**

The `tests/` folder contains several unit test scripts to evaluate different aspects of the model. Below are the commands to run these tests:

- **Test training process**:

  ```bash
  python tests/test_train.py
  ```

- **Test AUC scores**:

  ```bash
  python tests/test_auc.py
  ```

- **Visualize loss curves**:

  ```bash
  python tests/test_loss.py
  ```

- **Plot the confusion matrix**:

  ```bash
  python tests/test_confusion_matrix.py
  ```

- **Plot the precision-recall curve**:

  ```bash
  python tests/test_precision_recall.py
  ```

- **Plot the ROC curve**:
  ```bash
  python tests/test_roc.py
  ```

## Visualizations

This project includes several visualizations to better understand the dataset and track model performance:

1. **Fraud vs Non-Fraud Distribution**: Shows the imbalance in the dataset.
2. **Correlation Heatmap**: Displays correlations between PCA features (V1-V28).
3. **Training Time per Epoch**: Shows the time taken per epoch during training.
4. **AUC Score per Epoch**: Visualizes how the AUC score changes over the training process.
5. **Confusion Matrix**: Displays the model's true positives, false positives, true negatives, and false negatives.
6. **Precision-Recall Curve**: Important for analyzing model performance on imbalanced datasets.
7. **ROC Curve**: Shows the trade-off between the true positive rate and false positive rate.
8. **Loss per Epoch**: Tracks the model's loss during training.

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for more details.

## Acknowledgments

This project was conducted as part of a thesis at **Széchenyi István University**, supervised by **László Környei**. The credit card fraud dataset used in this project was provided by **Worldline** and the **Machine Learning Group** of ULB.

Please cite the following papers if you use this project:

- **Andrea Dal Pozzolo**, et al. "Calibrating Probability with Undersampling for Unbalanced Classification." In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
- **Dal Pozzolo, Andrea**, et al. "Learned lessons in credit card fraud detection from a practitioner perspective." Expert systems with applications, 41,10,4915-4928, 2014, Pergamon.
