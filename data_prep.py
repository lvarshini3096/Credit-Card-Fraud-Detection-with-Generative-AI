import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Helper function for data loading and preprocessing ---

def load_and_preprocess_data(file_path='Creditcard_dataset.csv'):
    """
    Loads, cleans, and scales the credit card fraud dataset.
    Returns: X (features), y (labels), and df_fraud (for GAN training)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None

    df.dropna(inplace=True)

    # Remove 'Time' column (as done in the notebook)
    df = df.drop(axis=1, columns='Time')

    # Scale 'Amount' column
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Split features and labels
    X = df.drop("Class", axis=1)
    y = df.Class

    # Separate fraud data for GAN training
    df_fraud = df[df.Class == 1].drop("Class", axis=1).copy()
    
    return X, y, df_fraud

if __name__ == '__main__':
    X, y, df_fraud = load_and_preprocess_data()
    
    if X is not None:
        print("Data Loading and Preprocessing Complete.")
        print(f"Full Data Shape (X, y): {X.shape}, {y.shape}")
        print(f"Fraudulent Samples (df_fraud): {df_fraud.shape}")
