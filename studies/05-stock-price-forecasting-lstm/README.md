# Stock Price Forecasting with Stacked LSTM Networks

A deep learning time-series forecasting study using stacked Long Short-Term Memory (LSTM) networks to model stock-price sequences from historical market data.

---

## 📌 Overview

This study explores sequence modeling for stock-price forecasting using a stacked LSTM neural network.

Historical stock-market variables are processed into fixed-length temporal sequences and used to predict the next closing-price value.

The workflow includes:

- Historical stock-data inspection
- Date parsing and time-index preparation
- Exploratory data visualization
- Synthetic sequence augmentation
- Feature scaling
- Sequential train/test splitting
- Sliding-window sequence generation
- Stacked LSTM development
- Model training and validation
- Prediction analysis
- Forecast visualization
- Next-step prediction from the latest sequence

---

## 📊 Dataset

The original dataset contains **253 historical observations** with the following columns:

| Variable | Description |
| --- | --- |
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Highest recorded price |
| `Low` | Lowest recorded price |
| `Close` | Closing price |
| `Volume` | Trading volume |

The `Date` column is converted to datetime format and then used as the DataFrame index.

The `Volume` column is removed before model development.

The forecasting workflow therefore uses four numerical variables:

```text
Close
High
Low
Open
```

The prediction target is:

```text
Close
```

---

## 🔎 Exploratory Analysis

The notebook includes several exploratory visualizations and descriptive analyses of the stock-price variables.

These include:

- Price evolution over time
- Box plots
- Pairwise feature relationships
- Feature distributions
- Correlation heatmap
- Descriptive statistics

These analyses provide an overview of the numerical structure and relationships within the historical market data.

---

## 🔄 Synthetic Sequence Augmentation

A synthetic augmentation procedure is applied to the original time-series variables using `statsmodels` and `ArmaProcess`.

A simulated sequence is generated and added independently to each of the four numerical stock-price variables:

- `Open`
- `High`
- `Low`
- `Close`

The generated observations are then concatenated with the original dataset.

This expands the data from:

- **253 original observations**
- to **506 total observations**

The augmented data is used as an exploratory strategy for increasing the amount of sequence data available for model training.

---

## ⚙️ Feature Scaling

The four model input variables are normalized using `MinMaxScaler`.

Input features:

```text
Close
High
Low
Open
```

Target:

```text
Close
```

The resulting arrays have the following dimensions:

```text
Features: (506, 4)
Target:   (506, 1)
```

---

## ✂️ Train/Test Split

A sequential 80/20 split is applied to the scaled arrays.

The resulting data sizes are:

| Split | Features | Target |
| --- | ---: | ---: |
| Training | 404 × 4 | 404 × 1 |
| Test | 102 × 4 | 102 × 1 |

The split preserves array order rather than randomly shuffling individual observations.

---

## ⏱️ Sequence Generation

The model uses a sliding-window approach to transform the numerical data into LSTM-compatible sequences.

The selected sequence length is:

```text
20 time steps
```

Each model input therefore contains:

```text
20 time steps × 4 features
```

After sequence generation:

| Dataset | Shape |
| --- | --- |
| `X_train` | `(384, 20, 4)` |
| `y_train` | `(384, 1)` |
| `X_test` | `(82, 20, 4)` |
| `y_test` | `(82, 1)` |

For each sequence, the model uses the previous 20 observations to predict the following closing-price value.

---

## 🧠 LSTM Architecture

The forecasting model is a stacked LSTM network:

```text
Input Sequence (20 × 4)
          ↓
LSTM (60, GELU, return_sequences=True)
          ↓
LSTM (40, GELU)
          ↓
Dropout (0.2)
          ↓
Dense (1)
```

Model size:

- **31,801 trainable parameters**

The first LSTM layer returns the complete temporal sequence to the second recurrent layer.

The second LSTM layer produces the encoded sequence representation used for the final regression prediction.

---

## ⚙️ Training Configuration

The network is compiled using:

- Optimizer: Adam
- Loss function: Mean Squared Error
- Monitoring metric: Mean Absolute Error
- Epochs: `700`
- Batch size: `8`

The test sequences are supplied as validation data during model training.

At the final recorded epoch, the normalized-space metrics are approximately:

| Metric | Training | Validation |
| --- | ---: | ---: |
| Loss | 0.0016 | 0.000252 |
| MAE | 0.0265 | 0.0117 |

The training history is visualized by comparing training and validation loss over the full optimization process.

---

## 📈 Forecast Visualization

After training, the model generates predictions for the test sequences.

Both predicted and reference values are transformed back to the original closing-price scale using the fitted scaler.

The notebook visualizes:

```text
Observed Closing Price
vs.
Predicted Closing Price
```

This provides a direct qualitative comparison between the modeled sequence output and the corresponding reference values.

---

## 🔮 Latest-Sequence Prediction

The final 20 observations are also passed through the trained network as a single sequence to demonstrate next-step prediction.

For the recorded notebook execution, the resulting predicted closing-price value is approximately:

```text
474.91
```

This value represents the model output for the specific final sequence and trained model state recorded in the notebook.

---

## ⚠️ Evaluation Notes

This notebook is an exploratory sequence-modeling study rather than a controlled financial forecasting benchmark.

Several implementation details should be considered when interpreting the recorded results:

- Feature scaling is fitted before the train/test split.
- The augmented dataset is created by concatenating the original and simulated sequences before splitting.
- The test sequences are also used as validation data during model training.
- The notebook's variable named `MAE` is calculated using `mean_squared_error`, so it should not be interpreted as an actual Mean Absolute Error value in the final evaluation table.
- The notebook calls `r2_score` with predicted and reference arrays in reversed argument order, so the displayed R² value is not used here as a definitive performance metric.

For these reasons, the study is best interpreted as a demonstration of LSTM-based time-series preprocessing, sequence construction, model development, and forecasting workflow.

---

## 💡 Key Concepts

This study demonstrates:

- Time-series forecasting
- Sequential data preprocessing
- Financial time-series visualization
- Synthetic sequence augmentation
- Min-max normalization
- Sequential train/test splitting
- Sliding-window generation
- Long Short-Term Memory networks
- Stacked LSTM architectures
- GELU activation
- Dropout regularization
- Adam optimization
- Regression with neural networks
- Sequence-to-value prediction
- Training-history analysis
- Inverse scaling
- Forecast visualization

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn

---

## 📁 Files

- [`stock-price-forecasting-lstm.ipynb`](./stock-price-forecasting-lstm.ipynb) — Complete implementation of preprocessing, sequence augmentation, window generation, LSTM training, prediction, and visualization.
- [`Download Data - STOCK_US_XNAS_META (1).csv`](./Download%20Data%20-%20STOCK_US_XNAS_META%20%281%29.csv) — Historical stock-market dataset used in the forecasting workflow.