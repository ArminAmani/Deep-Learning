# Deep Learning

A growing technical repository for deep learning implementations, experiments, studies, and applied projects across neural networks, computer vision, sequence modeling, representation learning, attention mechanisms, and related areas.

The repository is intended to evolve over time as new models, datasets, experiments, and engineering-oriented deep learning applications are developed and documented.

---

## 📌 Overview

This repository documents my ongoing work in deep learning through practical implementations and structured computational studies.

The current material covers several fundamental and applied topics, including:

- Feed-forward neural networks
- Regression with neural networks
- Multi-class image classification
- Regularization and generalization
- Convolutional neural networks
- Time-series modeling with LSTM networks
- Attention-based and Transformer-style architectures
- Imbalanced binary classification
- Model evaluation and diagnostic analysis

The repository is **not limited to the studies currently available**. Future additions may include larger projects, standalone implementations, comparative experiments, engineering applications, and more advanced deep learning architectures.

---

## 🎯 Objectives

The main goals of this repository are to:

- Implement deep learning models in practical computational settings
- Explore the behavior of different neural-network architectures
- Study training, regularization, optimization, and generalization
- Apply deep learning methods to structured, image, sequential, and time-series data
- Evaluate models using appropriate quantitative and visual diagnostics
- Document methodological limitations and implementation choices clearly
- Build a structured and continuously expanding deep learning portfolio
- Explore future applications of artificial intelligence in scientific and engineering problems

---

## 📂 Current Repository Structure

```text
Deep-Learning/
│
├── README.md
├── .gitignore
│
└── studies/
    ├── README.md
    │
    ├── 01-house-price-regression/
    │   ├── README.md
    │   ├── house-price-regression.ipynb
    │   └── House_Price_dataset.csv
    │
    ├── 02-cifar10-mlp-classification/
    │   ├── README.md
    │   └── cifar10-mlp-classification.ipynb
    │
    ├── 03-cifar10-regularization-and-generalization/
    │   ├── README.md
    │   └── cifar10-regularization-study.ipynb
    │
    ├── 04-cifar10-cnn-classification/
    │   ├── README.md
    │   └── cifar10-cnn-classification.ipynb
    │
    ├── 05-stock-price-forecasting-lstm/
    │   ├── README.md
    │   ├── stock-price-forecasting-lstm.ipynb
    │   └── Download Data - STOCK_US_XNAS_META (1).csv
    │
    └── 06-credit-card-fraud-transformer-classification/
        ├── README.md
        └── credit-card-fraud-transformer.ipynb
```

The structure will expand as additional work is added to the repository.

---

## 🧠 Current Deep Learning Studies

The `studies/` directory contains focused implementations exploring individual deep learning concepts, architectures, and modeling workflows.

| # | Study | Main Focus |
| --- | --- | --- |
| 01 | [House Price Regression](./studies/01-house-price-regression/) | Fully connected neural networks for tabular regression |
| 02 | [CIFAR-10 MLP Classification](./studies/02-cifar10-mlp-classification/) | Multi-layer perceptrons for image classification |
| 03 | [CIFAR-10 Regularization and Generalization](./studies/03-cifar10-regularization-and-generalization/) | L1/L2 regularization, early stopping, augmentation, and generalization |
| 04 | [CIFAR-10 CNN Classification](./studies/04-cifar10-cnn-classification/) | Convolutional neural networks, batch normalization, dropout, and image classification |
| 05 | [Stock Price Forecasting with LSTM](./studies/05-stock-price-forecasting-lstm/) | Sequential modeling and time-series forecasting with stacked LSTM networks |
| 06 | [Credit Card Fraud Classification with Transformer-Style Attention](./studies/06-credit-card-fraud-transformer-classification/) | Multi-head attention and Transformer-inspired modeling for imbalanced tabular classification |

Each study contains its own README with details about:

- Problem formulation
- Dataset
- Data preprocessing
- Model architecture
- Training configuration
- Evaluation
- Results
- Methodological considerations
- Technologies and key concepts

---

## 🔬 Topics Covered

### Neural Networks

- Dense neural networks
- Multi-layer perceptrons
- Regression networks
- Binary and multi-class classification
- Activation functions
- Network capacity and architecture design

### Computer Vision

- CIFAR-10 image classification
- Convolutional neural networks
- Image normalization
- Data augmentation
- Feature extraction through convolution
- Fully connected versus convolutional architectures

### Regularization & Generalization

- Dropout
- L1 regularization
- L2 regularization
- Early stopping
- Batch normalization
- Data augmentation
- Validation analysis
- Overfitting and generalization behavior

### Sequence Modeling

- Sequential data preparation
- Sliding-window generation
- Long Short-Term Memory networks
- Stacked recurrent architectures
- Time-series forecasting
- Sequence-to-value prediction

### Attention Mechanisms

- Self-attention
- Multi-head attention
- Feature embeddings
- Residual connections
- Layer normalization
- Feed-forward blocks
- Transformer-style architectures

### Model Evaluation

Depending on the problem, evaluation methods include:

- Mean Squared Error
- Mean Absolute Error
- R² analysis
- Classification accuracy
- Precision
- Recall
- F1-score
- Classification reports
- ROC analysis
- AUC
- Training and validation curves
- Prediction visualizations

---

## 🛠️ Technologies

The repository primarily uses:

### Programming & Development

- Python
- Jupyter Notebook
- Git
- GitHub

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pandas
- Scikit-learn

### Scientific & Statistical Tools

- Statsmodels

### Visualization

- Matplotlib
- Seaborn

Additional libraries and frameworks may be introduced as the repository expands.

---

## 🧩 Development Approach

The work in this repository generally follows a structured modeling workflow:

```text
Problem Definition
        ↓
Dataset Inspection
        ↓
Exploratory Analysis
        ↓
Data Preprocessing
        ↓
Model Design
        ↓
Training
        ↓
Validation
        ↓
Evaluation
        ↓
Diagnostic Analysis
        ↓
Documentation
```

The emphasis is not only on obtaining model outputs, but also on understanding:

- How the data is prepared
- Why a model architecture behaves in a particular way
- How training and validation performance differ
- Whether regularization improves generalization
- Which evaluation metrics are appropriate for the problem
- What methodological limitations affect interpretation of the results

---

## ⚠️ Methodological Transparency

The implementations in this repository are documented as computational studies and experiments.

Where relevant, individual study READMEs explicitly discuss issues such as:

- Data leakage
- Validation methodology
- Use of test data during model development
- Class imbalance
- Scaling methodology
- Non-isolated model comparisons
- Limitations of individual evaluation metrics
- Implementation-specific evaluation issues

Recorded numerical results should therefore be interpreted within the methodology of each individual study rather than as universal benchmark results.

This approach is intentional: documenting limitations is an important part of developing reliable machine-learning and deep-learning workflows.

---

## 📊 Data and Reproducibility

Datasets are handled according to their size and practical storage requirements.

Small datasets may be included directly when appropriate.

Large datasets may be excluded from Git tracking because of repository hosting limitations. In such cases, the corresponding study README provides information about the expected dataset and local file structure.

Some notebooks may therefore require locally available data before they can be executed from beginning to end.

---

## 🚀 Future Development

This repository is designed to grow beyond the current collection of studies.

Potential future additions may include:

- Standalone deep learning implementations
- Larger end-to-end projects
- Comparative architecture experiments
- Advanced convolutional architectures
- Transfer learning
- Representation learning
- Autoencoders
- Advanced recurrent architectures
- Transformer models
- Attention-based sequence models
- Scientific machine learning
- Physics-informed learning
- Deep learning for engineering data
- Surrogate modeling
- Reduced-order modeling
- Data-driven modeling of physical systems
- Aerospace and mechanical engineering applications
- Optimization and AI-assisted engineering workflows

New sections and directories will be introduced when corresponding implementations are added.

---

## 🧭 Repository Philosophy

The repository is developed around three principles:

**Implementation**  
Build and understand deep learning models through practical computational work.

**Evaluation**  
Analyze model behavior using suitable metrics, diagnostics, and validation procedures.

**Documentation**  
Present the methodology, results, and limitations of each implementation clearly and reproducibly.

The long-term objective is to develop a technically organized collection of deep learning work that connects data-driven methods with scientific and engineering applications.

---

## 📌 Status

This repository is actively evolving.

The current `studies/` collection represents the initial structured set of deep learning implementations. Additional experiments, projects, architectures, and engineering-oriented applications may be incorporated over time.

---

## 👤 Author

**Armin Amani**

M.Sc. Student in Aerospace Engineering (Aerodynamics)  
Sharif University of Technology

Areas of interest include:

- Aerospace Engineering
- Aerodynamics
- Scientific Computing
- Computational Engineering
- Machine Learning
- Deep Learning
- AI for Engineering

GitHub: [ArminAmani](https://github.com/ArminAmani)