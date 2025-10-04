# --- lr_pca_model.py ---
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from data_prep import load_and_preprocess_data # Import preprocessing helper

def train_and_evaluate_lr_pca(X, y, n_components=10):
    """
    Trains and evaluates Logistic Regression on PCA-transformed data.
    """
    print("--- Part 1: Logistic Regression on PCA Features ---")
    
    # 1. Split data (using original imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 2. Apply PCA
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Report explained variance
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
    print(f"PCA retains {explained_variance_ratio:.2%} of variance.")

    # 3. Train Logistic Regression Model
    print("\nTraining Logistic Regression...")
    lr_pca_model = LogisticRegression(solver='liblinear', random_state=42)
    lr_pca_model.fit(X_train_pca, y_train)

    # 4. Predict and Evaluate
    y_pred = lr_pca_model.predict(X_test_pca)
    y_prob = lr_pca_model.predict_proba(X_test_pca)[:, 1]

    print("\n--- Model Evaluation (LR on Imbalanced Data) ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC AUC Score:")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == '__main__':
    X, y, _ = load_and_preprocess_data()
    
    if X is not None:
        train_and_evaluate_lr_pca(X, y)
