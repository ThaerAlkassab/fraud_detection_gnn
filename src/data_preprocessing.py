# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Create a fraud label (as a placeholder for now; adjust based on actual logic)
    df['fraud'] = df['Transaction Description'].apply(lambda x: 1 if 'unusual' in x.lower() or 'debited' in x.lower() else 0)
    
    return df

def clean_financial_data(df, financial_columns, percentage_columns):
    """Clean dollar signs, commas, and percentage signs from financial and percentage data."""
    for col in financial_columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True).astype(float)

    for col in percentage_columns:
        df[col] = df[col].replace({'%': ''}, regex=True).astype(float)
    
    return df

# data_preprocessing.py

def preprocess_data(df):
    # List of financial and percentage features
    financial_features = ['Income Level', 'Account Balance', 'Deposits', 'Withdrawals', 'Transfers', 
                          'International Transfers', 'Investments', 'Loan Amount']
    percentage_features = ['Interest Rate']

    # Clean financial and percentage data
    df = clean_financial_data(df, financial_features, percentage_features)

    # Handle categorical data
    df['occupation_encoded'] = LabelEncoder().fit_transform(df['Occupation'])
    df['risk_tolerance_encoded'] = df['Risk Tolerance'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['loan_status_encoded'] = LabelEncoder().fit_transform(df['Loan Status'])
    df = pd.get_dummies(df, columns=['Investment Goals', 'Loan Purpose', 'Employment Status'])

    # Drop columns that are still of object type (e.g., Address, Transaction Description)
    df = df.select_dtypes(exclude=['object'])

    # Scale numeric features and rename them for clarity
    scaler = StandardScaler()
    df[['age_scaled', 'income_scaled', 'account_balance_scaled', 'loan_term_scaled', 'interest_rate_scaled']] = scaler.fit_transform(df[['Age', 'Income Level', 'Account Balance', 'Loan Term (Months)', 'Interest Rate']])

    # Drop the original columns after renaming
    df.drop(['Age', 'Income Level', 'Account Balance', 'Loan Term (Months)', 'Interest Rate'], axis=1, inplace=True)

    return df

if __name__ == "__main__":
    df = load_data('../data/5k.csv')
    df = preprocess_data(df)
    print(df.head())
