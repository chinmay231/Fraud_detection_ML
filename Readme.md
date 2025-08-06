# 💳 Fraud Detection Using Machine Learning

This repository demonstrates a complete machine learning pipeline for detecting fraudulent transactions using both **Logistic Regression** and **Random Forest** classifiers.

We focus on:
- Handling **class imbalance**, a critical issue in fraud detection.
- Comparing performance across baseline and balanced models.
- Visualizing confusion matrices and performance metrics.
- Delivering interpretable results suitable for real-world deployment.

---

## 📁 Dataset

The dataset used is `payment_fraud.csv`, which contains anonymized transaction records with the following features:

| Feature Name             | Description                                  |
|--------------------------|----------------------------------------------|
| `accountAgeDays`         | Age of the user account in days              |
| `numItems`               | Number of items in the transaction           |
| `localTime`              | Local time of transaction (as a float)       |
| `paymentMethod`          | Payment method used (categorical)            |
| `paymentMethodAgeDays`   | Time since payment method was added          |
| `label`                  | Target variable (1 = Fraud, 0 = Not Fraud)   |

---

## ⚙️ Pipeline Overview

1. **Data Preprocessing**
    - One-hot encode `paymentMethod` feature.
    - Split data using `train_test_split` with stratification.

2. **Models Implemented**
    - Logistic Regression (baseline)
    - Logistic Regression with `class_weight='balanced'`
    - Random Forest Classifier with `class_weight='balanced'`

3. **Model Evaluation**
    - Accuracy
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1-Score)
    - Heatmap visualization of confusion matrices

---

## 📊 Results Snapshot

| Model                   | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|------------------------|----------|-------------------|----------------|------------------|
| Logistic Regression    | 99.98%   | 0.98              | 1.00           | 0.99             |
| Balanced Logistic Reg. | 99.99%   | 0.99              | 1.00           | 1.00             |
| Random Forest          | 100%     | 1.00              | 1.00           | 1.00             |

The Random Forest model perfectly classified both fraud and non-fraud cases, with **zero false positives and zero false negatives** on the test set.

---

## 📈 Visualizations

Confusion matrices are plotted using `seaborn` for all models to compare their predictions clearly.

<p align="center">
  <img src="confusion_matrices.png" alt="Confusion Matrix Comparison" width="700">
</p>

---

## 💡 Key Takeaways

- **Class balancing matters**: It improved fraud precision from 0.98 to 0.99 in logistic regression.
- **Random Forest outperformed linear models** in this setting, especially for non-linear relationships.
- **Handling class imbalance is essential** in fraud detection pipelines — either via sampling or weighting.

---

## 📦 Dependencies

- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`

Install via pip:
```bash
pip install pandas scikit-learn matplotlib seaborn
