# CIFAR-10 Regularization & Generalization Study

A deep learning study focused on regularization, data augmentation, and generalization behavior in fully connected neural networks for CIFAR-10 image classification.

---

## 📌 Overview

This study investigates several strategies for controlling overfitting and improving generalization in multilayer perceptrons trained on the CIFAR-10 dataset.

The workflow compares multiple neural-network configurations under a common image-classification setting.

The study covers:

- CIFAR-10 image preprocessing
- Image normalization
- One-hot label encoding
- Feedforward neural networks
- Image data augmentation
- L1 regularization
- L2 regularization
- Combined L1 and L2 regularization
- Early stopping
- Combined regularization and early stopping
- Training and validation performance analysis

The objective is to compare how different regularization strategies influence learning behavior and validation performance.

---

## 📊 Dataset

The study uses the **CIFAR-10** dataset provided directly through TensorFlow / Keras.

| Split | Samples | Image Shape |
| --- | ---: | --- |
| Training set | 50,000 | 32 × 32 × 3 |
| Test set | 10,000 | 32 × 32 × 3 |

The dataset contains 10 image classes.

Before training:

- Image values are converted to floating-point format
- Pixel values are normalized by dividing by `255.0`
- Labels are transformed into one-hot encoded vectors

No external dataset file is required.

---

## 🖼️ Data Augmentation

Image augmentation is applied using `ImageDataGenerator`.

The transformations include:

- Rotation up to `10°`
- Horizontal shift up to `10%`
- Vertical shift up to `10%`
- Zoom variation up to `10%`

Horizontal flipping is disabled.

One augmented version of the original training set is generated and concatenated with the original data.

This expands the training data from:

- **50,000 original images**
- to **100,000 combined original and augmented samples**

The augmented image tensors are then flattened into vectors containing **3,072 features**.

---

## 🧠 Base Neural Network Architecture

The experiments use a fully connected architecture with the following hidden-layer structure:

```text
Input (3,072 features)
        ↓
Dense (256, ReLU)
        ↓
Dense (128, ReLU)
        ↓
Dense (64, ReLU)
        ↓
Dense (32, ReLU)
        ↓
Dense (16, ReLU)
        ↓
Dense (10, Softmax)
```

The common training configuration uses:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Maximum epochs: `30`
- Batch size: `64`

---

## 1️⃣ Baseline Model

The baseline model uses the fully connected architecture without explicit regularization or augmented training data.

Training is performed on the original CIFAR-10 training set with a `20%` validation split.

At the final epoch:

- Training accuracy: **0.5645**
- Validation accuracy: **0.4915**
- Validation loss: **1.4662**

Best recorded validation accuracy:

- **0.4977**

---

## 2️⃣ L1 Regularization

L1 kernel regularization is applied to each hidden Dense layer using:

```text
L1 coefficient = 0.0001
```

The model is trained using the combined original and augmented dataset.

At the final epoch:

- Training accuracy: **0.4658**
- Validation accuracy: **0.4672**
- Validation loss: **1.6344**

Best recorded validation accuracy:

- **0.4764**

---

## 3️⃣ L2 Regularization

L2 kernel regularization is applied to each hidden Dense layer using:

```text
L2 coefficient = 0.0001
```

At the final epoch:

- Training accuracy: **0.5286**
- Validation accuracy: **0.5163**
- Validation loss: **1.4458**

Best recorded validation accuracy:

- **0.5193**

Among the evaluated configurations, the L2-regularized model achieved the highest recorded validation accuracy.

---

## 4️⃣ Combined L1 + L2 Regularization

A combined L1/L2 regularizer is applied to the hidden layers using:

```text
L1 coefficient = 0.0001
L2 coefficient = 0.0001
```

At the final epoch:

- Training accuracy: **0.4627**
- Validation accuracy: **0.4669**
- Validation loss: **1.6442**

Best recorded validation accuracy:

- **0.4799**

---

## 5️⃣ Early Stopping

A separate model without kernel regularization is trained using an Early Stopping callback.

Configuration:

```text
Monitor: val_accuracy
Patience: 2
```

Training stops after **12 epochs** rather than reaching the maximum of 30 epochs.

At the final recorded epoch:

- Training accuracy: **0.5193**
- Validation accuracy: **0.5105**
- Validation loss: **1.3938**

Best recorded validation accuracy:

- **0.5157**

---

## 6️⃣ L1 Regularization + Early Stopping

The final configuration combines L1 regularization with Early Stopping.

Regularization:

```text
L1 coefficient = 0.0001
```

Early stopping configuration:

```text
Monitor: val_accuracy
Patience: 8
```

Training stops after **20 epochs**.

At the final recorded epoch:

- Training accuracy: **0.4466**
- Validation accuracy: **0.4618**
- Validation loss: **1.6294**

Best recorded validation accuracy:

- **0.4651**

---

## 📈 Model Comparison

| Configuration | Epochs Run | Final Validation Accuracy | Best Validation Accuracy |
| --- | ---: | ---: | ---: |
| Baseline | 30 | 0.4915 | 0.4977 |
| L1 Regularization | 30 | 0.4672 | 0.4764 |
| L2 Regularization | 30 | **0.5163** | **0.5193** |
| L1 + L2 Regularization | 30 | 0.4669 | 0.4799 |
| Early Stopping | 12 | 0.5105 | 0.5157 |
| L1 + Early Stopping | 20 | 0.4618 | 0.4651 |

For the configurations evaluated in the notebook, **L2 regularization produced the highest recorded validation accuracy**, followed closely by the Early Stopping model.

The results also show that different regularization strategies affect optimization and generalization differently rather than consistently improving performance.

---

## 📉 Training Analysis

The notebook compares validation accuracy and validation loss across all model configurations.

Detailed plots are also generated for each model to compare:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These visualizations help examine convergence behavior, generalization gaps, and the effect of different regularization strategies during training.

---

## ⚠️ Evaluation Note

The baseline model uses a validation split taken from the training data.

For the regularized and Early Stopping configurations, the notebook passes the CIFAR-10 test split directly as `validation_data` during training.

Therefore, the reported results for those configurations should be interpreted as **validation-monitoring results on the provided test split**, rather than performance from a completely untouched final test evaluation.

---

## 💡 Key Concepts

This study demonstrates:

- Image classification with multilayer perceptrons
- CIFAR-10 preprocessing
- Image normalization
- One-hot encoding
- Data augmentation
- Fully connected neural networks
- L1 regularization
- L2 regularization
- Combined L1/L2 regularization
- Early stopping
- Overfitting analysis
- Generalization analysis
- Training and validation monitoring
- Comparative model evaluation

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## 📁 File

- [`cifar10-regularization-study.ipynb`](./cifar10-regularization-study.ipynb) — Complete implementation of data preprocessing, augmentation, regularization experiments, Early Stopping, model training, and comparative analysis.