# House Price Regression with Neural Networks

A deep learning regression study focused on predicting property prices from selected numerical housing and geographic features using fully connected neural networks.

---

## 📌 Overview

This study explores the application of feedforward neural networks to a house-price regression problem.

The workflow covers:

- Dataset inspection
- Numerical feature selection
- Outlier analysis and filtering
- Train/test splitting
- Feature standardization
- Fully connected neural-network development
- Dropout-based regularization
- Model training
- Regression performance evaluation
- Training-history visualization

Two neural-network configurations are implemented and compared on a held-out test set.

---

## 📊 Dataset

The original dataset contains:

- **168,446 observations**
- **20 attributes**

The dataset includes information related to property characteristics, geographic location, listing details, and price.

For the regression workflow, the following numerical variables are selected:

| Variable | Role |
| --- | --- |
| `latitude` | Input feature |
| `longitude` | Input feature |
| `baths` | Input feature |
| `bedrooms` | Input feature |
| `Area Size` | Input feature |
| `price` | Prediction target |

---

## ⚙️ Data Preparation

The preprocessing workflow includes:

1. Inspection of the dataset structure and missing values
2. Selection of numerical variables relevant to the regression task
3. Outlier analysis using the Interquartile Range (IQR)
4. Removal of observations outside the defined IQR bounds for the selected input features
5. Separation of input features and target values
6. Train/test splitting
7. Feature standardization using `StandardScaler`

After outlier filtering, the modeling dataset contains:

- **161,994 observations**
- **5 input features**

The data is divided into:

- **113,395 training samples**
- **48,599 test samples**

A **70/30 train-test split** is used.

The scaler is fitted on the training features and then applied to both the training and test sets.

---

## 🧠 Neural Network Models

### Model 1 — Baseline Feedforward Network

The first model uses a fully connected architecture:

```text
Input (5 features)
        ↓
Dense (5, ReLU)
        ↓
Dense (64, ReLU)
        ↓
Dense (16, ReLU)
        ↓
Dense (1)
```

Training configuration:

- Optimizer: Adam
- Learning rate: `0.1`
- Loss function: Mean Squared Error
- Monitoring metric: Mean Absolute Error
- Epochs: `300`
- Batch size: `16`
- Validation split: `0.3`

---

### Model 2 — Feedforward Network with Dropout

The second model introduces Dropout regularization within a deeper fully connected architecture:

```text
Input (5 features)
        ↓
Dense (5, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (32, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (16, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (1)
```

Training configuration:

- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: Mean Squared Error
- Monitoring metric: Mean Absolute Error
- Epochs: `500`
- Batch size: `32`
- Validation split: `0.3`

---

## 📈 Evaluation

Model performance is evaluated on the held-out test set using the coefficient of determination (**R² score**).

| Model | Test R² |
| --- | ---: |
| Baseline Feedforward Network | **0.3751** |
| Feedforward Network with Dropout | **0.2920** |

For the configurations evaluated in this study, the baseline feedforward network achieved the higher test-set R² score.

It is important to note that the two models differ not only in the use of Dropout, but also in architecture, learning rate, batch size, and number of training epochs. Therefore, the comparison represents the performance of two different neural-network configurations rather than a controlled measurement of the isolated effect of Dropout.

---

## 📉 Training Analysis

Training and validation histories are recorded during model fitting and visualized to examine the optimization behavior of both neural-network configurations.

The recorded histories support analysis of:

- Training progression
- Validation behavior
- Differences between the two model configurations
- Potential generalization limitations

---

## 💡 Key Concepts

This study demonstrates:

- Neural networks for regression
- Tabular-data preprocessing
- Numerical feature selection
- IQR-based outlier filtering
- Train/test splitting
- Feature standardization
- Fully connected neural networks
- ReLU activation
- Dropout regularization
- Adam optimization
- Mean Squared Error loss
- Mean Absolute Error monitoring
- R²-based regression evaluation
- Training-history analysis

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

- [`house-price-regression.ipynb`](./house-price-regression.ipynb) — Complete implementation of data preprocessing, neural-network development, training, evaluation, and visualization.
- [`House_Price_dataset.csv`](./House_Price_dataset.csv) — Dataset used for the house-price regression study.