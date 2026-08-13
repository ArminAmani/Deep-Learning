# CIFAR-10 Image Classification with Convolutional Neural Networks

A deep learning image-classification study using convolutional neural networks to classify CIFAR-10 images and compare two CNN configurations with different activation and regularization strategies.

---

## 📌 Overview

This study explores convolutional neural networks for multiclass image classification on the CIFAR-10 dataset.

Unlike fully connected image-classification approaches that flatten the input immediately, the models in this study preserve the spatial structure of the images and use convolutional layers to learn hierarchical visual features.

The workflow includes:

- CIFAR-10 dataset loading
- Image visualization
- Pixel-value normalization
- One-hot label encoding
- Image data augmentation
- Convolutional neural-network development
- Batch normalization
- Max pooling
- Dropout regularization
- L2 weight regularization
- ReLU and ELU activation functions
- Adaptive learning-rate and early-stopping callbacks
- Training and validation analysis
- Test-set evaluation
- Classification-report analysis

Two CNN configurations are implemented and compared.

---

## 📊 Dataset

The study uses the **CIFAR-10** dataset provided through TensorFlow / Keras.

| Split | Samples | Image Shape |
| --- | ---: | --- |
| Training set | 50,000 | 32 × 32 × 3 |
| Test set | 10,000 | 32 × 32 × 3 |

Each image belongs to one of 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## ⚙️ Data Preparation

### Image Normalization

The image arrays are converted to floating-point values and normalized by dividing the original pixel values by `255.0`.

This scales the image data to the range:

```text
0.0 → 1.0
```

### Label Encoding

The class labels are converted into one-hot encoded vectors with 10 output categories using TensorFlow / Keras utilities.

---

## 🖼️ Data Augmentation

Image augmentation is applied to the training set using `ImageDataGenerator`.

The configured transformations include:

- Rotation up to `10°`
- Horizontal shift up to `10%`
- Vertical shift up to `10%`
- Zoom variation up to `10%`
- Horizontal flipping disabled

One augmented version of the complete training set is generated and concatenated with the original images.

The resulting training dataset contains:

- **50,000 original images**
- **50,000 augmented images**
- **100,000 total training samples**

The corresponding labels are concatenated in the same manner.

---

## 🧠 Model 1 — CNN with ReLU

The first model uses convolutional feature-extraction blocks with ReLU activation, Batch Normalization, Max Pooling, and Dropout.

Architecture:

```text
Input Image (32 × 32 × 3)
        ↓
Conv2D (32, 3×3, ReLU)
        ↓
Batch Normalization
        ↓
Conv2D (32, 3×3, ReLU)
        ↓
Batch Normalization
        ↓
MaxPooling2D (2×2)
        ↓
Dropout (0.3)
        ↓
Conv2D (64, 3×3, ReLU)
        ↓
Batch Normalization
        ↓
Conv2D (64, 3×3, ReLU)
        ↓
Batch Normalization
        ↓
MaxPooling2D (2×2)
        ↓
Dropout (0.5)
        ↓
Flatten
        ↓
Dense (64, ReLU)
        ↓
Batch Normalization
        ↓
Dropout (0.5)
        ↓
Dense (10, Softmax)
```

Training configuration:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Epochs: `10`
- Batch size: `64`

For this model, the CIFAR-10 test split is supplied as validation data during training.

At the final epoch:

- Training accuracy: **0.7665**
- Validation accuracy: **0.8017**
- Validation loss: **0.5927**

---

## 🧠 Model 2 — CNN with ELU and L2 Regularization

The second CNN retains the general convolutional structure while modifying the activation and regularization strategy.

The main changes are:

- ELU activation instead of ReLU
- L2 kernel regularization in convolutional layers
- Learning-rate reduction callback
- Early-stopping callback

The L2 coefficient used in the convolutional layers is:

```text
0.0001
```

Architecture:

```text
Input Image (32 × 32 × 3)
        ↓
Conv2D (32, 3×3, ELU, L2)
        ↓
Batch Normalization
        ↓
Conv2D (32, 3×3, ELU, L2)
        ↓
Batch Normalization
        ↓
MaxPooling2D (2×2)
        ↓
Dropout (0.3)
        ↓
Conv2D (64, 3×3, ELU, L2)
        ↓
Batch Normalization
        ↓
Conv2D (64, 3×3, ELU, L2)
        ↓
Batch Normalization
        ↓
MaxPooling2D (2×2)
        ↓
Dropout (0.5)
        ↓
Flatten
        ↓
Dense (64, ELU)
        ↓
Batch Normalization
        ↓
Dropout (0.5)
        ↓
Dense (10, Softmax)
```

Training configuration:

- Optimizer: Adam
- Loss function: Categorical Cross-Entropy
- Metric: Accuracy
- Maximum epochs: `20`
- Batch size: `64`
- Validation split: `0.2`
- Shuffle: Enabled

---

## 🔄 Training Callbacks

Two callbacks are defined for Model 2.

### Reduce Learning Rate on Plateau

```text
Monitor: val_accuracy
Factor: 0.2
Patience: 5
Minimum learning rate: 0.0001
```

### Early Stopping

```text
Monitor: val_loss
Patience: 5
```

In the recorded training run, Model 2 completed all **20 epochs**.

At the final epoch:

- Training accuracy: **0.7932**
- Validation accuracy: **0.7887**
- Validation loss: **0.6741**

The highest recorded validation accuracy during this training run was approximately:

- **0.7973**

---

## 📈 Test-Set Evaluation

Both trained models are evaluated on the CIFAR-10 test set containing **10,000 images**.

| Model | Test Loss | Test Accuracy |
| --- | ---: | ---: |
| CNN with ReLU | **0.5927** | 0.8017 |
| CNN with ELU + L2 | 0.6135 | **0.8138** |

The second model achieved the higher test accuracy, while the first model produced the lower test loss.

Therefore, the comparison illustrates that model quality should not be interpreted using a single metric alone.

---

## 📋 Classification Performance

Predicted Softmax probabilities are converted into class indices using `argmax`.

Classification reports are then generated using:

- Precision
- Recall
- F1-score
- Support
- Overall accuracy
- Macro averages
- Weighted averages

### Model 1

Overall test accuracy:

- **0.80**

Macro-average F1-score:

- **0.80**

### Model 2

Overall test accuracy:

- **0.81**

Macro-average F1-score:

- **0.81**

The results show a modest overall improvement for the second CNN configuration on the evaluated test set.

---

## 📉 Training Analysis

The notebook visualizes validation accuracy and validation loss across both model configurations.

Detailed learning curves are also generated separately for each model, comparing:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These visualizations support analysis of convergence behavior and generalization during CNN training.

---

## ⚠️ Evaluation Note

The two models use different validation procedures.

For Model 1, the CIFAR-10 test set is supplied directly as `validation_data` during training and is later used again for final evaluation.

For Model 2, validation data is created using a `20%` split from the augmented training dataset, while the CIFAR-10 test set is reserved for evaluation after training.

Because the test set is monitored during the training of Model 1, its reported test performance should not be interpreted as an estimate from a completely untouched final test set.

The comparison should therefore be treated as an analysis of the recorded model configurations rather than a strictly controlled benchmark.

---

## 💡 Key Concepts

This study demonstrates:

- Convolutional neural networks
- CIFAR-10 image classification
- Image normalization
- One-hot label encoding
- Image data augmentation
- Convolutional feature extraction
- ReLU activation
- ELU activation
- Batch normalization
- Max pooling
- Dropout regularization
- L2 weight regularization
- Adam optimization
- Categorical Cross-Entropy
- Learning-rate scheduling
- Early stopping
- Softmax multiclass classification
- Precision, recall, and F1-score evaluation
- Training and validation analysis
- Comparative CNN evaluation

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

- [`cifar10-cnn-classification.ipynb`](./cifar10-cnn-classification.ipynb) — Complete implementation of CIFAR-10 preprocessing, image augmentation, CNN development, training, evaluation, and comparative analysis.