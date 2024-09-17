# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Create a fraud label (as a placeholder for now; adjust based on actual logic)
    df['fraud'] = df['Transaction Description'].apply(lambda x: 1 if 'unusual' in x.lower() or 'debited' in x.lower() else 0)
    
    return df

def clean_financial_data(df, columns):
    """Clean dollar signs and commas from financial data."""
    for col in columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)
    return df

def preprocess_data(df):
    # List of financial features that need cleaning
    financial_features = ['Income Level', 'Account Balance', 'Deposits', 'Withdrawals', 'Transfers', 
                          'International Transfers', 'Investments', 'Loan Amount']

    # Clean financial data
    df = clean_financial_data(df, financial_features)

    # Handle categorical data
    df['occupation_encoded'] = LabelEncoder().fit_transform(df['Occupation'])
    df['risk_tolerance_encoded'] = df['Risk Tolerance'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['loan_status_encoded'] = LabelEncoder().fit_transform(df['Loan Status'])
    df = pd.get_dummies(df, columns=['Investment Goals', 'Loan Purpose', 'Employment Status'])

    # Scale numeric features
    scaler = StandardScaler()
    df[financial_features + ['Age', 'Loan Term (Months)', 'Interest Rate']] = scaler.fit_transform(df[financial_features + ['Age', 'Loan Term (Months)', 'Interest Rate']])

    return df

if __name__ == "__main__":
    df = load_data('../data/5k.csv')
    df = preprocess_data(df)
    print(df.head())
