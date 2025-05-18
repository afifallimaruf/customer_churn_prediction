import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

def preprocess_data(file_path, output_path):
    # load dataset
    df = pd.read_csv(file_path)
    
    # delete customerID
    df.drop('customerID', axis=1, inplace=True)
    df.drop('gender', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(df.info())
    # strategy for fill missing values
    cat_imputer = SimpleImputer(strategy='most_frequent')
    # Handle missing value
    for col in df.columns:
        if df[col].dtype == 'float64' and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        if df[col].dtype == 'object' and df[col].isna().any():
            df[col] = cat_imputer.fit_transform(df)
    
    # encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    num_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_features:
        df[col] = df[col]/max(df[col])

    print(df.head())

    # save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    preprocess_data('dataset/extracted_data/Telco-Customer-Churn.csv', 'dataset/processed/processed_data.csv')
