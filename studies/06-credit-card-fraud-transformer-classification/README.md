# Credit Card Fraud Classification with Transformer-Style Attention

A deep learning study exploring Transformer-style self-attention for binary classification of highly imbalanced credit-card transaction data.

---

## 📌 Overview

This study investigates the application of a Transformer-inspired attention architecture to credit-card fraud classification.

The model treats the numerical transaction features as an ordered feature representation, projects them into an embedding space, and processes them using multi-head self-attention, residual connections, layer normalization, and a feed-forward network.

The workflow includes:

- Dataset inspection
- Missing-value analysis
- Exploratory class-distribution analysis
- Feature-distribution visualization
- Feature standardization
- Train/test splitting
- Transformer-style model construction
- Multi-head self-attention
- Residual connections
- Layer normalization
- Feed-forward processing
- Binary classification
- Training-history analysis
- ROC-curve evaluation
- Discussion of evaluation limitations

---

## 📊 Dataset

The dataset contains:

- **284,807 transactions**
- **31 original columns**
- **30 numerical input-related variables**
- One binary target variable: `Class`

The class distribution is highly imbalanced:

| Class | Meaning | Samples |
| --- | --- | ---: |
| `0` | Non-fraudulent transaction | 284,315 |
| `1` | Fraudulent transaction | 492 |

Fraudulent transactions therefore represent only a very small fraction of the complete dataset.

The original variables include:

```text
Time
V1 ... V28
Amount
Class
```

The `Time` column is removed before model development.

The final model input therefore contains **29 numerical features**:

```text
V1 ... V28
Amount
```

Target:

```text
Class
```

---

## ⚙️ Data Preparation

### Missing Values

The dataset is inspected for missing values.

No missing observations are detected in the recorded dataset.

---

### Feature Selection

The `Time` variable is removed before training.

The remaining 29 numerical variables are used as model inputs.

---

### Standardization

The input variables are standardized using `StandardScaler`.

The transformation is applied before the train/test split in the recorded notebook.

---

### Train/Test Split

The standardized dataset is divided using a random 70/30 split:

| Split | Samples |
| --- | ---: |
| Training set | 199,364 |
| Test set | 85,443 |

The split uses:

```text
random_state = 100
```

Because the target distribution is highly imbalanced and the split is not explicitly stratified, class balance should be considered when interpreting the resulting evaluation.

---

## 🔎 Exploratory Analysis

The notebook contains exploratory visualizations for examining the dataset and class structure.

These include:

- Transaction class distribution with respect to time
- Feature-wise density comparisons between the two classes
- Histograms of standardized model inputs

These visualizations help illustrate differences between fraudulent and non-fraudulent transaction distributions.

---

## 🧠 Transformer-Style Architecture

The model uses a Transformer-inspired self-attention block for tabular binary classification.

The 29 numerical input features are first reshaped and projected into a 64-dimensional representation.

Architecture:

```text
Input (29 features)
        ↓
Reshape (29 × 1)
        ↓
Dense Projection (64)
        ↓
Multi-Head Self-Attention
    8 attention heads
        ↓
Dropout (0.1)
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Feed-Forward Network
    Dense (128, ReLU)
        ↓
    Dense (64)
        ↓
Dropout (0.1)
        ↓
Residual Connection
        ↓
Layer Normalization
        ↓
Global Average Pooling
        ↓
Dropout (0.1)
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.1)
        ↓
Dense (1, Sigmoid)
```

Model size:

- **153,857 trainable parameters**

---

## ⚙️ Model Hyperparameters

The main architecture parameters are:

| Parameter | Value |
| --- | ---: |
| Input length | 29 |
| Embedding dimension | 64 |
| Attention heads | 8 |
| Feed-forward dimension | 128 |
| Attention/FFN dropout | 0.1 |

---

## 🏋️ Training Configuration

The model is compiled using:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss function: Binary Cross-Entropy
- Metric: Accuracy
- Epochs: `10`
- Batch size: `32`

The test split is supplied as validation data during model training.

At the final recorded epoch:

| Metric | Value |
| --- | ---: |
| Training loss | 0.0057 |
| Training accuracy | 0.9987 |
| Validation loss | 0.0060 |
| Validation accuracy | 0.9986 |

---

## 📉 Training Analysis

The notebook visualizes:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These learning curves provide a view of optimization behavior over the 10 recorded training epochs.

---

## 📈 ROC Analysis

The trained model generates continuous fraud-probability predictions for the test set.

These probability scores are used to construct a Receiver Operating Characteristic (ROC) curve.

The notebook calculates:

- False Positive Rate
- True Positive Rate
- ROC curve
- Area Under the Curve (AUC)

The ROC visualization provides an evaluation perspective beyond raw classification accuracy.

---

## ⚠️ Evaluation Considerations

This dataset is **extremely imbalanced**, with only 492 fraudulent transactions among 284,807 total observations.

For this reason, accuracy alone is not sufficient for assessing fraud-detection quality. A classifier can obtain very high accuracy largely by predicting the majority non-fraud class.

Additional metrics such as the following are particularly important for this problem:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion matrix

The notebook attempts to generate a classification report; however, the recorded implementation applies `argmax` to the 29-dimensional input features and to a single-output sigmoid prediction.

That operation does not produce the intended binary fraud labels.

Therefore, the classification report generated by that cell is **not treated as a valid model-performance result in this documentation**.

A correct binary classification evaluation would instead threshold the sigmoid probability output and compare the resulting `0/1` predictions directly against `y_test`.

---

## ⚠️ Methodological Notes

Several implementation details should be considered when interpreting the study:

- `StandardScaler` is fitted to the complete feature matrix before the train/test split rather than only on training data.
- The test split is also used as validation data during training.
- The target classes are highly imbalanced.
- Accuracy is therefore not an adequate standalone evaluation metric.
- The notebook defines a positional-encoding function, but positional encoding is not applied in the final model.
- The architecture is best described as **Transformer-style or attention-based** rather than a complete canonical Transformer implementation.
- The final classification-report cell does not correctly construct binary predictions.

For these reasons, the study is presented primarily as an exploration of attention-based neural architectures for tabular fraud classification rather than as a production-ready fraud-detection benchmark.

---

## 💡 Key Concepts

This study demonstrates:

- Binary classification
- Highly imbalanced datasets
- Credit-card fraud detection
- Tabular-data preprocessing
- Standardization
- Multi-head self-attention
- Transformer-style architectures
- Feature embedding
- Residual connections
- Layer normalization
- Feed-forward neural networks
- Global average pooling
- Dropout regularization
- Sigmoid classification
- Binary Cross-Entropy
- Adam optimization
- ROC analysis
- Evaluation of imbalanced classification problems

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Files

- [`credit-card-fraud-transformer.ipynb`](./credit-card-fraud-transformer.ipynb) — Complete implementation of preprocessing, Transformer-style model construction, training, prediction, and evaluation.

### Dataset

The dataset file is intentionally **not stored in this Git repository** because its size exceeds GitHub's standard per-file limit.

The notebook expects the dataset to be available locally under the filename:

```text
creditcard.csv
```

To reproduce the study, place `creditcard.csv` in the same directory as the notebook before execution.