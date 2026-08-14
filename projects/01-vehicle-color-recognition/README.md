# Color Detection & Vehicle Color Recognition

A computer vision project combining a lightweight rule-based color detector with a deep learning approach for multi-class vehicle color recognition using an ImageNet-pretrained ResNet50 network.

The project presents two complementary approaches to visual color analysis:

1. **Real-time color identification with OpenCV and HSV thresholding**
2. **Vehicle color classification with end-to-end fine-tuning of ResNet50**

---

## 📌 Overview

Color recognition can be approached from fundamentally different perspectives.

The first implementation in this project uses classical computer vision. A webcam frame is converted from BGR to HSV, the pixel at the center of the image is sampled, and manually defined HSV rules are used to assign a color label in real time.

The second implementation treats vehicle color recognition as a supervised deep learning problem. Vehicle images from 15 color categories are used to fine-tune an ImageNet-pretrained ResNet50 model for multi-class classification.

Together, the two notebooks demonstrate the contrast between:

```text
Rule-Based Computer Vision
            vs.
Learned Visual Representation
```

---

## 📂 Project Structure

```text
01-vehicle-color-recognition/
│
├── README.md
│
└── notebooks/
    ├── color-detection-opencv.ipynb
    └── vehicle-color-recognition-resnet50.ipynb
```

---

# 1. Real-Time Color Detection with OpenCV

## 🎯 Objective

The first notebook implements a lightweight real-time color identification system using a webcam and manually defined HSV thresholds.

No machine-learning model is used in this part.

Instead, color identification is based directly on the Hue, Saturation, and Value components of the pixel located at the center of each captured video frame.

Notebook:

[`notebooks/color-detection-opencv.ipynb`](./notebooks/color-detection-opencv.ipynb)

---

## ⚙️ Processing Pipeline

The implementation follows this workflow:

```text
Webcam Input
     ↓
Capture Video Frame
     ↓
Convert BGR → HSV
     ↓
Locate Center Pixel
     ↓
Extract H, S, V Values
     ↓
Apply Rule-Based Thresholds
     ↓
Assign Color Label
     ↓
Display Result in Real Time
```

The webcam is initialized with a target resolution of:

```text
1280 × 720
```

For each frame, the image center is calculated as:

```text
cx = width / 2
cy = height / 2
```

The HSV value of that single center pixel is then used for color identification.

---

## 🎨 Supported Color Labels

The rule-based classifier can return the following labels:

```text
Black
Gray
White
Red
Orange
Yellow
Light Yellow
Light Green
Green
Cyan
Light Blue
Blue
Purple
Pink
```

An `Undefined` result is used as a fallback when the sampled HSV value does not satisfy the defined conditions.

Black, gray, and white are primarily separated using saturation and value thresholds, while chromatic colors are determined using predefined hue intervals.

---

## 🖥️ Real-Time Visualization

During execution, the notebook:

- Captures frames continuously from the default camera
- Converts each frame to HSV
- Samples the center pixel
- Determines the corresponding color name
- Displays the detected color on the video frame
- Marks the sampled center location
- Terminates when the `ESC` key is pressed

This implementation is intentionally simple and demonstrates direct rule-based color interpretation rather than object-level or region-level color recognition.

---

## ⚠️ OpenCV Method Limitations

The OpenCV implementation should be interpreted as a center-pixel color detector.

Its result depends directly on:

- The exact pixel located at the center of the frame
- Camera characteristics
- Illumination conditions
- Reflections and shadows
- Saturation and brightness
- The manually selected HSV thresholds

It does not perform:

- Object detection
- Image segmentation
- Dominant-color estimation over an entire object
- Learned color classification

The approach is therefore most useful as a compact demonstration of real-time HSV-based color identification.

---

# 2. Vehicle Color Recognition with ResNet50

## 🎯 Objective

The second notebook formulates vehicle color recognition as a 15-class image-classification problem.

A ResNet50 convolutional neural network initialized with ImageNet weights is adapted to classify vehicle images according to their visible color.

Notebook:

[`notebooks/vehicle-color-recognition-resnet50.ipynb`](./notebooks/vehicle-color-recognition-resnet50.ipynb)

---

## 📊 Dataset

The dataset used in the recorded notebook is organized into separate training, validation, and test directories.

The implementation expects the following Google Drive structure:

```text
/content/drive/MyDrive/train
/content/drive/MyDrive/val
/content/drive/MyDrive/test
```

The processed splits contain:

| Split | Images | Classes |
| --- | ---: | ---: |
| Training | 7,091 | 15 |
| Validation | 1,490 | 15 |
| Test | 1,511 | 15 |
| **Total** | **10,092** | **15** |

The dataset itself is not included in this repository.

The source or formal dataset name is not specified directly inside the notebook, so this repository documents the dataset according to the directory structure and class labels used by the implementation.

---

## 🎨 Vehicle Color Classes

The 15 target classes are:

```text
beige
black
blue
brown
gold
green
grey
orange
pink
purple
red
silver
tan
white
yellow
```

The training distribution is not perfectly uniform across classes.

For example, the recorded training split contains:

```text
blue: 737 images
red: 625 images
gold: 193 images
```

This variation in class frequency is relevant when interpreting class-level performance.

---

## 🖼️ Image Preprocessing

All images are resized to:

```text
64 × 64 × 3
```

Pixel values are rescaled using:

```text
1 / 255
```

The training generator also applies data augmentation using:

- Horizontal flipping
- Width shifting with `width_shift_range = 0.5`

The recorded configuration does not apply:

- Vertical flipping
- Rotation augmentation

Validation and test images are rescaled but are not subjected to the training augmentation pipeline.

---

## 🧠 Model Architecture

The model uses:

```python
ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(64, 64, 3),
    pooling="max"
)
```

The original ImageNet classification head is removed and replaced by a custom classification head.

The resulting architecture can be summarized as:

```text
Input Image
   64 × 64 × 3
        ↓
ResNet50 Backbone
ImageNet Initialization
include_top = False
pooling = "max"
        ↓
Batch Normalization
        ↓
Dense (256)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Dropout (0.5)
        ↓
Dense (15)
        ↓
Softmax
```

No explicit freezing of the ResNet50 backbone is performed in the notebook.

The model therefore performs **end-to-end fine-tuning** rather than using ResNet50 solely as a frozen feature extractor.

---

## 🔢 Model Size

The recorded model contains:

| Parameter Type | Count |
| --- | ---: |
| Total parameters | 24,125,327 |
| Trainable parameters | 24,067,599 |
| Non-trainable parameters | 57,728 |

Most of the ResNet50 network therefore remains trainable during optimization.

---

## ⚙️ Training Configuration

The model is compiled with:

| Setting | Value |
| --- | --- |
| Optimizer | Adamax |
| Learning rate | `0.0001` |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |
| Batch size | `32` |
| Maximum epochs | `50` |

Three callbacks are used during training:

### Early Stopping

```text
Monitor: val_loss
Patience: 10
Restore best weights: True
```

### Model Checkpoint

The best recorded model is saved as:

```text
ResNet50-model.best.keras
```

### CSV Logger

Training history is also written to:

```text
training.log
```

The recorded run terminated after epoch 32.

---

## ⚠️ Training Log Consideration

The stored training output contains repeated warnings indicating that the generator input ran out of data during some epochs.

This is associated with the recorded combination of Keras generators and manually specified:

```text
steps_per_epoch
validation_steps
```

As a consequence, some individual training and validation values in the saved learning history are not directly comparable across epochs.

For example, the recorded log contains an isolated validation accuracy of `1.0000` during one epoch, but that value should **not** be interpreted as a reliable overall model-performance result.

For this reason, this documentation emphasizes evaluation on the complete test generator rather than isolated validation values from the training log.

---

## 🧪 Test Evaluation

The test generator contains:

```text
1,511 images
15 classes
batch_size = 1
shuffle = False
```

Predicted class probabilities are produced for the complete test set.

The predicted class is obtained with:

```text
argmax(probabilities)
```

and compared directly with the reference class labels.

The notebook calculates:

- Overall categorical accuracy
- Per-class precision
- Per-class recall
- Per-class F-score
- Class support

It also visualizes 40 randomly selected test images with predicted and true labels.

---

## 📈 Test Performance

The recorded overall test accuracy is:

```text
83.32%
```

Per-class results are:

| Class | Precision | Recall | F-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| Green | 0.9746 | 0.9829 | **0.9787** | 117 |
| Blue | 0.9500 | 0.9560 | **0.9530** | 159 |
| Yellow | 0.9580 | 0.9421 | **0.9500** | 121 |
| Purple | 0.9720 | 0.9043 | **0.9369** | 115 |
| Red | 0.8452 | 0.9776 | **0.9066** | 134 |
| Pink | 0.9375 | 0.8738 | **0.9045** | 103 |
| Orange | 0.8991 | 0.8750 | **0.8869** | 112 |
| White | 0.8387 | 0.9176 | **0.8764** | 85 |
| Black | 0.8043 | 0.8605 | **0.8315** | 86 |
| Brown | 0.8302 | 0.7928 | **0.8111** | 111 |
| Grey | 0.6893 | 0.7802 | **0.7320** | 91 |
| Silver | 0.7692 | 0.5405 | **0.6349** | 74 |
| Gold | 0.7000 | 0.5122 | **0.5915** | 41 |
| Tan | 0.5263 | 0.5063 | **0.5161** | 79 |
| Beige | 0.4526 | 0.5181 | **0.4831** | 83 |

---

## 🔎 Result Interpretation

The model performs particularly well for several visually distinctive color categories.

The strongest recorded F-scores are obtained for:

```text
Green   → 0.9787
Blue    → 0.9530
Yellow  → 0.9500
Purple  → 0.9369
Red     → 0.9066
```

The most challenging categories are:

```text
Beige   → 0.4831
Tan     → 0.5161
Gold    → 0.5915
Silver  → 0.6349
```

These results suggest that visually similar, neutral, or metallic color categories are more difficult for the recorded model to separate than several highly saturated colors.

This observation is consistent with the class-level metrics, but a more detailed error analysis would require additional tools such as a confusion matrix.

---

## 🔬 What This Project Demonstrates

The project brings together two different computer-vision paradigms.

### Classical Computer Vision

- Webcam acquisition
- BGR-to-HSV conversion
- Pixel-level color sampling
- HSV thresholding
- Rule-based color classification
- Real-time visualization with OpenCV

### Deep Learning

- Multi-class image classification
- Convolutional neural networks
- ResNet50
- ImageNet initialization
- End-to-end fine-tuning
- Data augmentation
- Batch normalization
- Dropout
- Softmax classification
- Early stopping
- Model checkpointing
- Per-class performance evaluation

---

## 🛠️ Technologies

- Python
- Jupyter Notebook
- OpenCV
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab / Google Drive workflow

---

## ▶️ Reproducibility

### OpenCV Notebook

The OpenCV implementation requires:

- A local Python environment
- OpenCV
- NumPy
- Access to a webcam
- A graphical environment capable of displaying OpenCV windows

Run:

```text
notebooks/color-detection-opencv.ipynb
```

and place the desired surface or object so that the target color is located at the center of the camera frame.

---

### ResNet50 Notebook

The deep learning notebook expects the dataset to be available in:

```text
/content/drive/MyDrive/train
/content/drive/MyDrive/val
/content/drive/MyDrive/test
```

with one subdirectory per color class.

Example:

```text
train/
├── beige/
├── black/
├── blue/
├── brown/
├── ...
└── yellow/
```

The same 15-class directory organization is expected for validation and test data.

Dataset files and trained model checkpoints are not stored in this repository.

---

## ⚠️ Methodological Notes

The recorded results should be interpreted within the implementation used in the notebooks.

Important considerations include:

- The OpenCV approach evaluates only the center pixel rather than an entire object or image region.
- HSV thresholds are manually defined and may require adjustment under different lighting conditions or cameras.
- The ResNet50 input resolution is `64 × 64`, substantially smaller than the resolution commonly associated with ImageNet-trained ResNet models.
- The ResNet50 backbone is not explicitly frozen, so the complete network is fine-tuned.
- The training class distribution is not uniform.
- The recorded Keras training log contains generator exhaustion warnings.
- Some individual validation metrics are therefore not treated as definitive performance indicators.
- Test accuracy and per-class metrics are used as the primary recorded evaluation results.
- The notebook does not currently compute a confusion matrix or ROC analysis.

These points are documented to keep the reported results technically transparent.

---

## 🚀 Future Development

Potential extensions include:

- Region-based rather than center-pixel OpenCV color estimation
- Automatic HSV calibration
- Dominant-color extraction
- Object detection before color classification
- Vehicle localization followed by color recognition
- Confusion-matrix analysis
- Class-balancing strategies
- Higher input resolutions
- Alternative pretrained CNN architectures
- More controlled transfer-learning and fine-tuning experiments
- Improved data augmentation
- Real-time deep learning inference
- Comparison between classical and learned color-recognition pipelines

---

## 📁 Files

### [`color-detection-opencv.ipynb`](./notebooks/color-detection-opencv.ipynb)

Real-time center-pixel color identification using webcam input, HSV conversion, manually defined color thresholds, and OpenCV visualization.

### [`vehicle-color-recognition-resnet50.ipynb`](./notebooks/vehicle-color-recognition-resnet50.ipynb)

Fifteen-class vehicle color recognition using an ImageNet-initialized ResNet50 network, data augmentation, end-to-end fine-tuning, and test-set evaluation.

---

## 📌 Project Status

This project is part of the evolving `projects/` section of the broader **Deep-Learning** repository.

The current implementation combines classical computer vision and deep learning approaches to color recognition and may be extended with additional architectures, evaluation methods, and deployment-oriented workflows in the future.