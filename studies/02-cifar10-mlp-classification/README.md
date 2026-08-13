# CIFAR-10 Classification with Feedforward Neural Networks

A deep learning image-classification study using fully connected neural networks to classify CIFAR-10 images after flattening the image representation into numerical feature vectors.

---

## 📌 Overview

This study explores multiclass image classification using feedforward neural networks on the CIFAR-10 dataset.

Rather than using convolutional layers, the images are flattened and passed directly through fully connected neural-network architectures.

The workflow includes:

- CIFAR-10 dataset loading
- Image normalization
- One-hot encoding of class labels
- Image flattening
- Feedforward neural-network development
- Comparison of two network architectures
- Model training and validation
- Classification-based evaluation
- Training-history visualization

Two multilayer perceptron configurations with different network capacities are implemented and compared.

---

## 📊 Dataset

The study uses the **CIFAR-10** image dataset provided through TensorFlow / Keras.

The loaded dataset contains:

| Split | Samples | Image Shape |
| --- | ---: | --- |
| Training set | 50,000 | 32 × 32 × 3 |
| Test set | 10,000 | 32 × 32 × 3 |

Each image is represented by:

- Width: `32`
- Height: `32`
- Channels: `3`
- Number of classes: `10`

The original class labels are transformed into one-hot encoded vectors before model training.

---

## ⚙️ Data Preparation

The preprocessing workflow includes two main transformations.

### Image Normalization

Pixel values are normalized from their original integer range by dividing them by `255.0`.

This transforms the input images into normalized floating-point representations suitable for neural-network training.

### Label Encoding

The class labels are converted into one-hot encoded vectors with 10 output categories using:

```python
tf.keras.utils.to_categorical(...)
```

No external dataset file is required because CIFAR-10 is loaded directly through the TensorFlow / Keras dataset interface.

---

## 🧠 Neural Network Models

Both models use fully connected layers rather than convolutional layers.

The `32 × 32 × 3` image tensors are first flattened into vectors containing **3,072 input values**.

Leaky ReLU activations are used throughout the hidden layers, followed by a Softmax output layer for multiclass classification.

---

### Model 1 — Baseline MLP

Architecture:

```text
Input Image (32 × 32 × 3)
          ↓
Flatten (3,072 features)
          ↓
Dense (128, Leaky ReLU)
          ↓
Dense (64, Leaky ReLU)
          ↓
Dense (32, Leaky ReLU)
          ↓
Dense (16, Leaky ReLU)
          ↓
Dense (10, Softmax)
```

Total parameters:

- **404,378 trainable parameters**

Training configuration:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Epochs: `100`
- Batch size: `16`
- Validation split: `0.3`

---

### Model 2 — Higher-Capacity MLP

The second model increases network depth and parameter capacity.

Architecture:

```text
Input Image (32 × 32 × 3)
          ↓
Flatten (3,072 features)
          ↓
Dense (256, Leaky ReLU)
          ↓
Dense (128, Leaky ReLU)
          ↓
Dense (64, Leaky ReLU)
          ↓
Dense (32, Leaky ReLU)
          ↓
Dense (16, Leaky ReLU)
          ↓
Dense (10, Softmax)
```

Total parameters:

- **830,618 trainable parameters**

Training configuration:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Epochs: `100`
- Batch size: `16`
- Validation split: `0.3`

---

## 📈 Evaluation

Both models are evaluated on the held-out CIFAR-10 test set containing **10,000 images**.

Predicted class probabilities are converted to class indices using `argmax`, and performance is analyzed with classification reports containing precision, recall, F1-score, and accuracy.

### Overall Test Performance

| Model | Test Accuracy | Macro F1 |
| --- | ---: | ---: |
| Baseline MLP | **0.48** | **0.47** |
| Higher-Capacity MLP | **0.48** | **0.47** |

Both models achieved approximately the same overall test accuracy despite the substantially larger parameter count of Model 2.

---

## 📉 Training Behavior

The training histories reveal a noticeable difference between training and validation performance toward the end of training.

At the final epoch:

| Model | Training Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| Baseline MLP | 0.6716 | 0.4647 |
| Higher-Capacity MLP | 0.7331 | 0.4761 |

The higher-capacity model achieves greater training accuracy, while validation and test performance remain close to those of the smaller network.

This indicates that simply increasing the size of a fully connected architecture does not necessarily produce a corresponding improvement in generalization performance for this image-classification task.

---

## 📊 Training Visualization

For both models, the notebook visualizes:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

These plots provide a direct comparison between optimization progress on the training data and model behavior on the validation subset.

---

## 💡 Key Concepts

This study demonstrates:

- Multiclass image classification
- CIFAR-10 data loading
- Image normalization
- One-hot label encoding
- Image flattening
- Multilayer perceptrons
- Fully connected neural networks
- Leaky ReLU activation
- Softmax classification
- Adam optimization
- Categorical Cross-Entropy
- Train/validation monitoring
- Precision, recall, and F1-score evaluation
- Model-capacity comparison
- Generalization analysis

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

## 📁 File

- [`cifar10-mlp-classification.ipynb`](./cifar10-mlp-classification.ipynb) — Complete implementation of CIFAR-10 preprocessing, feedforward neural-network development, training, evaluation, and visualization.