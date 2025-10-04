# Credit-Card-Fraud-Detection-with-Generative-AI

# 💳 Credit Card Fraud Detection with Generative AI (GANs)

This project tackles the critical problem of **credit card fraud detection** by leveraging **Generative Adversarial Networks (GANs)** to address the severe **data imbalance** inherent in financial transaction datasets.

The core goal was to generate high-quality synthetic fraudulent transactions, effectively balancing the dataset to improve the final model's ability to catch real-world fraud.

---

## 💡 Project Highlights & Results

The original dataset is extremely imbalanced (only **0.172%** fraud cases). The primary business goal was to maximize **Recall** for the fraud class.

| Metric | Initial LR (Imbalanced Data) | Final LR (GAN-Augmented Data) | Impact of GANs |
| :--- | :---: | :---: | :--- |
| **Fraud Recall (Class 1)** | 0.51 | **0.85** | **+66% improvement** in catching fraud (Sensitivity). |
| **Fraud Precision (Class 1)**| 0.83 | 0.81 | Maintained high precision. |
| **F1-Score (Class 1)** | 0.63 | **0.83** | Significant increase in model effectiveness. |
| **ROC AUC Score** | 0.9382 | 0.8861 | A drop is expected when testing on a balanced, more challenging distribution. |

---

## 🔬 Methodology and Process (Brief)

1.  **Initial Benchmark:**
    * **Data Prep:** Scaled the `Amount` feature.
    * **Modeling:** Applied **PCA** (10 components) followed by **Logistic Regression** on the original, imbalanced data.
    * **Result:** Established a low baseline **Fraud Recall of 0.51**.
2.  **Generative Augmentation (GAN):**
    * Trained a **Generative Adversarial Network** exclusively on the **492 real fraudulent transactions**.
    * The **Generator** learned the latent structure of the fraud data.
    * Used the trained Generator to create **~284,000 synthetic fraud samples**, resulting in a 50/50 balanced dataset.
3.  **Final Model Evaluation:**
    * Retrained the **LR on PCA** model using the new, **balanced dataset**.
    * **Result:** Achieved a successful and balanced **Fraud Recall of 0.85**.

---

## 🛠️ Setup and Execution

### 1. Data Source

The project uses the anonymized credit card transaction dataset from Kaggle.

* **Dataset Link:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Action:** Download `creditcard.csv` and rename it to **`Creditcard_dataset.csv`** in the root directory.

### 2. Dependencies

Install all required dependencies using the `requirements.txt` file (see contents below):

```bash
pip install -r requirements.txt
```

## 🏃 How to Run the Code
The project is structured into three main scripts to show the progression from initial data handling to the final augmented model. Run these commands sequentially from your terminal:

### Run Preprocessing (data_prep.py):
Loads, scales, and prepares the data for both the LR and GAN models.

```bash
python data_prep.py
```
### Run Initial Benchmark (lr_pca_model.py):
Trains and evaluates the Logistic Regression model on the original, imbalanced data (Phase 1).

```bash
python lr_pca_model.py
```
### Run GAN Training & Final Model (gan_model.py):
Builds and trains the GAN, generates synthetic data, creates the balanced dataset, and finally trains/evaluates the LR model on the GAN-augmented data (Phases 2 & 3).

### This step is computationally intensive and may take time depending on your hardware
```bash
python gan_model.py
```
