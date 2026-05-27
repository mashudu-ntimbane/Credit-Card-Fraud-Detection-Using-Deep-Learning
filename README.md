# Credit Card Fraud Detection Using Deep Learning

## Project Overview

This project investigates the use of deep learning techniques for detecting fraudulent credit card transactions in a highly imbalanced financial dataset. The study evaluates and compares multiple deep learning approaches, including supervised, unsupervised, sequential, and generative models, to determine the most effective method for fraud detection.

The primary challenge addressed in this project is the extreme class imbalance present in real-world fraud detection datasets, where fraudulent transactions represent only a small fraction of all transactions. Various imbalance-handling techniques, including class weighting, anomaly detection, and synthetic data generation, were implemented and evaluated.

---
## Project Highlights

✔ Developed and evaluated four deep learning architectures for fraud detection

✔ Addressed extreme class imbalance (0.172% fraud rate) using class weighting and GAN-based synthetic oversampling

✔ Achieved a PR-AUC of 0.7228 using a GAN-augmented classifier

✔ Reduced false positives by approximately 98.9% compared to the baseline DNN

✔ Evaluated supervised, unsupervised, sequential, and generative learning approaches

✔ Built using TensorFlow, Keras, Scikit-Learn, and Python

---

## Results and Visualizations

### Model Performance Comparison

<img width="1189" height="590" alt="Matrices comparison bar chart plots" src="https://github.com/user-attachments/assets/1b95301c-4f69-4c31-83c2-d6d7af3037b1" />


### Precision-Recall Curve

<img width="861" height="706" alt="Precision-Recall plot" src="https://github.com/user-attachments/assets/e4693063-2922-4766-961e-1c9c288b4634" />


### ROC Curve
<img width="861" height="706" alt="ROC curve" src="https://github.com/user-attachments/assets/cdc89ffe-7f10-4c71-b358-5a33f6ecfd87" />

### Confusion Matrix (Best Model)

![Confusion Matrix](figures/Confusion Matrix plots.png)

### GAN-Based Class Augmentation

<img width="1349" height="1189" alt="Confusion Matrix plots" src="https://github.com/user-attachments/assets/1521ab3a-9a7b-4b15-93b3-1cba79135e64" />


### DNN Training History
<img width="1790" height="985" alt="DNN training history" src="https://github.com/user-attachments/assets/7fb9bb95-6610-4775-be9c-348d0109763f" />

### Autoencoder Training History
<img width="1790" height="985" alt="Autoencoder training history" src="https://github.com/user-attachments/assets/e6fef745-ea87-4ae5-b98a-336605865049" />
<img width="1005" height="552" alt="Autoencoder error distribution" src="https://github.com/user-attachments/assets/7f567f25-26b1-4db2-9122-5874ea3215e0" />

### LSTM Training History
<img width="1790" height="985" alt="LSTM training history" src="https://github.com/user-attachments/assets/a95c1858-6efc-4ea6-9264-143f1df8f736" />

### GAN Training History
<img width="1788" height="985" alt="GAN training history" src="https://github.com/user-attachments/assets/ded7e4d5-e868-4eb4-ac64-e726d8129644" />

---

## Key Results

| Metric | Best Model (GAN + Classifier) |
|----------|----------|
| Precision | 71% |
| Recall | 81% |
| F1-Score | 0.76 |
| ROC-AUC | 0.9769 |
| PR-AUC | 0.7228 |
| False Positives | 31 |
| Fraud Detection Rate | 81% |

---

## Why This Project Matters

Financial institutions process millions of transactions daily, making fraud detection a critical challenge. Traditional rule-based systems often struggle to adapt to evolving fraud patterns and generate large numbers of false positives.

This project demonstrates how deep learning and synthetic data generation can be leveraged to improve fraud detection performance while significantly reducing false alarms. The results highlight the effectiveness of GAN-based augmentation for learning from highly imbalanced datasets and provide practical insights for real-world fraud detection systems.

## Objectives

The objectives of this project were to:

- Develop robust fraud detection models capable of identifying fraudulent transactions.
- Compare the performance of multiple deep learning architectures.
- Address severe class imbalance using advanced machine learning techniques.
- Evaluate model effectiveness using metrics appropriate for imbalanced classification.
- Recommend a suitable approach for real-world fraud detection systems.

---

## Dataset

The project uses the Credit Card Fraud Detection dataset originally collected through a research collaboration between Worldline and the Machine Learning Group of Université Libre de Bruxelles (ULB).

### Dataset Characteristics

| Feature | Description |
|----------|----------|
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 |
| Legitimate Transactions | 284,315 |
| Fraud Rate | 0.172% |
| Imbalance Ratio | 1:578 |
| Features | 30 Numerical Features |

### Features

- **V1–V28:** PCA-transformed anonymized features
- **Time:** Seconds elapsed since the first transaction
- **Amount:** Transaction amount
- **Class:** Target variable (0 = Legitimate, 1 = Fraud)

---

## Data Preprocessing

The following preprocessing steps were performed:

### Data Quality Assessment

- Removed duplicate records
- Verified absence of missing values
- Confirmed all features were numerical

### Feature Engineering

#### Time Feature

- Converted transaction time into hour-of-day representation
- Applied sine and cosine transformations to capture cyclical patterns

#### Amount Feature

- Scaled using RobustScaler to reduce the influence of outliers

### Data Splitting

The dataset was split using stratified sampling:

- Training Set: 60%
- Validation Set: 20%
- Test Set: 20%

### Class Imbalance Handling

Several techniques were employed to address the severe class imbalance:

- Class weighting
- Autoencoder-based anomaly detection
- GAN-based synthetic fraud generation
- PR-AUC focused evaluation

---

## Models Implemented

### 1. Deep Neural Network (DNN)

A supervised learning baseline consisting of:

- Dense Layers
- Batch Normalization
- Dropout Regularization
- Class-weighted training

### 2. Autoencoder

An unsupervised anomaly detection model trained exclusively on legitimate transactions.

Fraud detection was performed using reconstruction error thresholds.

### 3. Long Short-Term Memory (LSTM)

A sequential deep learning architecture designed to identify temporal transaction patterns using sliding-window sequences.

### 4. GAN-Augmented Classifier

A Generative Adversarial Network (GAN) was used to generate synthetic fraudulent transactions to improve minority-class representation.

The generated samples were then used to train a fraud classification model.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Google Colab

---

## Experimental Results

### Final Test Set Performance

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---------|---------:|---------:|---------:|---------:|---------:|
| DNN Baseline | 0.03 | 0.87 | 0.06 | 0.9671 | 0.6279 |
| Autoencoder | 0.00 | 0.93 | 0.01 | - | - |
| LSTM | 0.00 | 0.29 | 0.00 | 0.5128 | 0.0019 |
| GAN + Classifier | 0.71 | 0.81 | 0.76 | 0.9769 | 0.7228 |

---

## Best Performing Model

### GAN-Augmented Classifier

The GAN-Augmented Classifier achieved the best overall performance:

- Precision: 71%
- Recall: 81%
- F1-Score: 0.76
- ROC-AUC: 0.9769
- PR-AUC: 0.7228

Key benefits include:

- Strong balance between precision and recall
- Significant reduction in false positives
- Improved minority-class representation
- Better suitability for real-world deployment

---

## Key Findings

- Class imbalance significantly impacts fraud detection performance.
- Accuracy is not a reliable metric for highly imbalanced datasets.
- Precision-Recall AUC provides a more informative evaluation than accuracy.
- Class weighting improves minority-class detection.
- Autoencoders achieve high recall but suffer from poor precision.
- LSTM models are ineffective when meaningful temporal patterns are absent.
- GAN-generated synthetic fraud samples improve classifier performance.
- GAN augmentation achieved the best balance between fraud detection and false positive reduction.

---

## Project Structure

```text
credit-card-fraud-detection-deep-learning/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_DNN_Model.ipynb
│   ├── 03_Autoencoder_Model.ipynb
│   ├── 04_LSTM_Model.ipynb
│   └── 05_GAN_Classifier.ipynb
│
├── figures/
│   ├── dnn_training_history.png
│   ├── autoencoder_loss.png
│   ├── reconstruction_error.png
│   ├── lstm_training_history.png
│   ├── gan_training_history.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   └── threshold_optimization.png
│
├── report/
│   └── Credit_Card_Fraud_Detection_Report.pdf
│
└── data/
    └── dataset_information.txt
```

---

## Future Work

Future improvements may include:

- Ensemble learning approaches
- Transformer-based architectures
- Graph Neural Networks
- Advanced GAN architectures
- Hybrid machine learning and rule-based systems

---

## Author

**MASHUDU NTIMBANE,**
**PATEL MOHAMEDSUHAIL,**
**NEMANGWELA UNARINE, and<img width="1189" height="590" alt="Matrices comparison bar chart plots" src="https://github.com/user-attachments/assets/a85aaed3-3faf-4d9c-a477-490c42307e21" />
**
**SHABANE WAMASHUDU**

Areas of Interest:

- Artificial Intelligence
- Machine Learning
- Data Science
- Software Development

---
