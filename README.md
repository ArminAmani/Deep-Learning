# Deep Learning

A growing technical repository for deep learning implementations, computational studies, experiments, and applied projects across neural networks, computer vision, sequence modeling, attention mechanisms, representation learning, and related areas.

The repository is designed to evolve over time as new models, datasets, experiments, and engineering-oriented applications are developed and documented.

---

## 📌 Overview

This repository documents my ongoing work in deep learning through both focused computational studies and broader applied projects.

The current material covers several fundamental and applied topics, including:

- Feed-forward neural networks
- Neural-network regression
- Multi-class image classification
- Regularization and generalization
- Convolutional neural networks
- Transfer learning and fine-tuning
- Computer vision with OpenCV
- ResNet-based image classification
- Time-series modeling with LSTM networks
- Attention-based and Transformer-style architectures
- Imbalanced binary classification
- Model evaluation and diagnostic analysis

The repository is **not limited to the material currently available**. Future additions may include larger end-to-end projects, standalone implementations, comparative experiments, advanced architectures, scientific machine learning, and engineering-oriented AI applications.

---

## 🎯 Objectives

The main goals of this repository are to:

- Implement deep learning models in practical computational settings
- Explore the behavior of different neural-network architectures
- Study training, regularization, optimization, and generalization
- Apply deep learning methods to structured, image, sequential, and time-series data
- Explore classical and deep learning approaches to computer vision
- Evaluate models using appropriate quantitative and visual diagnostics
- Document methodological choices and limitations clearly
- Build a structured and continuously expanding technical portfolio
- Explore applications of artificial intelligence in scientific and engineering problems

---

## 🗂️ Repository Organization

The repository currently uses two primary sections:

| Directory | Purpose |
| --- | --- |
| [`studies/`](./studies/) | Focused computational studies exploring specific deep learning concepts, architectures, and modeling workflows |
| [`projects/`](./projects/) | Broader applied projects organized around practical problems and potentially combining multiple computational approaches |

Additional sections may be introduced as the repository expands.

---

## 📂 Current Repository Structure

```text
Deep-Learning/
│
├── README.md
├── .gitignore
│
├── projects/
│   ├── README.md
│   │
│   └── 01-vehicle-color-recognition/
│       ├── README.md
│       └── notebooks/
│           ├── color-detection-opencv.ipynb
│           └── vehicle-color-recognition-resnet50.ipynb
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

This structure represents the current state of the repository and will expand as additional work is added.

---

## 🚀 Current Applied Projects

The [`projects/`](./projects/) directory contains broader application-oriented work that may combine multiple techniques within a single problem.

| # | Project | Main Focus |
| --- | --- | --- |
| 01 | [Vehicle Color Recognition](./projects/01-vehicle-color-recognition/) | Rule-based real-time color detection with OpenCV and deep learning-based vehicle color classification using an ImageNet-initialized ResNet50 |

### Vehicle Color Recognition

This project explores two complementary computer-vision approaches:

1. **Rule-based real-time color identification**
   - Webcam-based image acquisition
   - BGR-to-HSV conversion
   - Center-pixel sampling
   - HSV threshold-based color identification
   - OpenCV visualization

2. **Deep learning-based vehicle color classification**
   - 15-class image classification
   - ResNet50
   - ImageNet initialization
   - End-to-end fine-tuning
   - Data augmentation
   - Per-class precision, recall, and F-score analysis

The project demonstrates the contrast between classical rule-based computer vision and learned visual representations.

---

## 🧠 Current Deep Learning Studies

The [`studies/`](./studies/) directory contains focused implementations exploring individual deep learning concepts, architectures, and modeling workflows.

| # | Study | Main Focus |
| --- | --- | --- |
| 01 | [House Price Regression](./studies/01-house-price-regression/) | Fully connected neural networks for tabular regression |
| 02 | [CIFAR-10 MLP Classification](./studies/02-cifar10-mlp-classification/) | Multi-layer perceptrons for image classification |
| 03 | [CIFAR-10 Regularization and Generalization](./studies/03-cifar10-regularization-and-generalization/) | L1/L2 regularization, early stopping, augmentation, and generalization |
| 04 | [CIFAR-10 CNN Classification](./studies/04-cifar10-cnn-classification/) | Convolutional neural networks, batch normalization, dropout, and image classification |
| 05 | [Stock Price Forecasting with LSTM](./studies/05-stock-price-forecasting-lstm/) | Sequential modeling and time-series forecasting with stacked LSTM networks |
| 06 | [Credit Card Fraud Classification with Transformer-Style Attention](./studies/06-credit-card-fraud-transformer-classification/) | Multi-head attention and Transformer-inspired modeling for imbalanced tabular classification |

Individual study and project READMEs provide more detailed information about:

- Problem formulation
- Dataset
- Data preprocessing
- Architecture and implementation
- Training configuration
- Evaluation methodology
- Results
- Methodological considerations
- Reproducibility
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

- Real-time color identification
- OpenCV
- HSV color-space processing
- CIFAR-10 image classification
- Convolutional neural networks
- ResNet50
- ImageNet initialization
- Transfer learning and fine-tuning
- Image normalization
- Data augmentation
- Learned visual representations
- Fully connected versus convolutional image models

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
- Class-level performance analysis

---

## 🛠️ Technologies

The repository primarily uses:

### Programming & Development

- Python
- Jupyter Notebook
- Git
- GitHub
- Google Colab

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Data Processing & Machine Learning

- NumPy
- Pandas
- Scikit-learn

### Scientific & Statistical Tools

- Statsmodels

### Visualization

- Matplotlib
- Seaborn

Additional libraries, frameworks, and computational tools may be introduced as the repository expands.

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
Model / Method Design
        ↓
Training or Implementation
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
- Why a particular method or architecture is selected
- How model architecture affects behavior
- How training and validation performance differ
- Whether regularization improves generalization
- Which evaluation metrics are appropriate for the problem
- What methodological limitations affect interpretation of the results
- How classical computational approaches compare with learned models when relevant

---

## ⚠️ Methodological Transparency

The implementations in this repository are documented as computational studies, experiments, and applied projects.

Where relevant, individual READMEs explicitly discuss issues such as:

- Data leakage
- Validation methodology
- Use of test data during model development
- Class imbalance
- Scaling methodology
- Non-isolated model comparisons
- Training-pipeline limitations
- Dataset availability
- Limitations of individual evaluation metrics
- Implementation-specific evaluation issues

Recorded numerical results should therefore be interpreted within the methodology of each individual study or project rather than as universal benchmark results.

Documenting these limitations is an important part of developing reliable machine-learning and deep-learning workflows.

---

## 📊 Data and Reproducibility

Datasets are handled according to their size, source, and practical storage requirements.

Small datasets may be included directly when appropriate.

Large datasets or externally organized image collections may be excluded from Git tracking because of storage limitations or dataset-specific requirements.

In these cases, the corresponding project or study README describes:

- Expected dataset organization
- Required local paths where applicable
- Relevant preprocessing
- Execution requirements

Some notebooks therefore require locally available or externally mounted data before they can be executed from beginning to end.

---

## 🚀 Future Development

This repository is designed to grow beyond the current collection of studies and projects.

Potential future additions may include:

- Standalone deep learning implementations
- Larger end-to-end projects
- Comparative architecture experiments
- Advanced convolutional architectures
- Transfer-learning studies
- Representation learning
- Autoencoders
- Advanced recurrent architectures
- Transformer models
- Attention-based sequence models
- Object detection
- Advanced computer-vision workflows
- Scientific machine learning
- Physics-informed learning
- Deep learning for engineering data
- Surrogate modeling
- Reduced-order modeling
- Data-driven modeling of physical systems
- Aerospace and mechanical engineering applications
- Optimization and AI-assisted engineering workflows

New directories and organizational sections will be introduced when corresponding implementations are added.

---

## 🧭 Repository Philosophy

The repository is developed around three principles:

**Implementation**  
Build and understand deep learning models and computational methods through practical work.

**Evaluation**  
Analyze model behavior using suitable metrics, diagnostics, and validation procedures.

**Documentation**  
Present the methodology, results, limitations, and reproducibility considerations of each implementation clearly.

The long-term objective is to develop a technically organized collection of deep learning and AI work that connects data-driven methods with scientific and engineering applications.

---

## 📌 Status

This repository is actively evolving.

The current content includes:

- A structured collection of focused deep learning studies
- An expanding section of applied projects
- Implementations spanning structured data, image data, time series, computer vision, recurrent networks, convolutional networks, and attention-based architectures

Additional studies, projects, experiments, architectures, and engineering-oriented applications may be incorporated over time.

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