# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Creating a fraud label by checking for "unusual" or large amounts in the description
    df['fraud'] = df['Transaction Description'].apply(lambda x: 1 if 'unusual' in x.lower() or 'debited' in x.lower() else 0)
    
    return df

def preprocess_data(df):
    # Handling categorical data
    df['occupation_encoded'] = LabelEncoder().fit_transform(df['Occupation'])
    df['risk_tolerance_encoded'] = df['Risk Tolerance'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['loan_status_encoded'] = LabelEncoder().fit_transform(df['Loan Status'])
    df = pd.get_dummies(df, columns=['Investment Goals', 'Loan Purpose', 'Employment Status'])

    # Scaling numeric features
    scaler = StandardScaler()
    financial_features = ['Age', 'Income Level', 'Account Balance', 'Deposits', 'Withdrawals', 'Transfers', 
                          'International Transfers', 'Investments', 'Loan Amount', 'Loan Term (Months)', 'Interest Rate']
    df[financial_features] = scaler.fit_transform(df[financial_features])
    
    return df

if __name__ == "__main__":
    df = load_data('../data/5k.csv')
    df = preprocess_data(df)
    print(df.head())
