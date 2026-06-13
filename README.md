# Power Grid Anomaly Detection

Deep learning-based power grid anomaly detection using sequence modeling, reconstruction error analysis, and classification for grid event monitoring.

![Application Interface](images/application_interface.png)

## Overview

This project presents a deep learning pipeline for detecting anomalies in power grid data using sequence-based modeling. It combines an LSTM autoencoder for anomaly-sensitive feature learning with a downstream classifier for fault category prediction, and also includes a Streamlit interface for interactive analysis and visualization.

## Key Features

- LSTM-based sequence modeling for power grid anomaly detection.
- Autoencoder-driven latent feature learning from multivariate grid signals.
- - Two-stage pipeline: anomaly detection with an LSTM autoencoder, followed by anomaly-type classification with an LSTM classifier.
- Streamlit application for interactive fault analysis and visualization.
- Modular project structure with saved model weights and deployment files.

## Dataset and Classes

The project works on multivariate power system measurements and frames anomaly detection as a multi-class learning problem.

The notebook defines the following operating classes:

- Normal
- Voltage Anomaly
- Frequency Anomaly
- Power Factor Anomaly
- Combined Anomaly

![Anomaly Reference](images/anomaly_reference.png)

![Class Distribution](images/class_distribution.png)

### Dataset Summary

The notebook uses the **PowerGridSense** dataset and loads a multivariate grid monitoring table with **10,000 time-stamped records**.

**Input features used for modeling**

- Voltage (V)
- Current (A)
- Power (kW)
- Frequency (Hz)
- Power Factor

The following metadata columns are present in the raw dataset but removed before model training:

- `SensorID`
- `Location`

### Grid Condition Labels

The project defines five operating conditions:

- **0 — Normal**
- **1 — Voltage Anomaly**
- **2 — Frequency Anomaly**
- **3 — Power Factor Anomaly**
- **4 — Combined Anomaly**

## Methodology

The workflow begins with preprocessing and organizing the power grid signals into sequence windows suitable for recurrent modeling.

An LSTM autoencoder is used to learn sequence representations, after which learned features are used for downstream classification of grid conditions.

This design allows the project to combine temporal pattern learning with anomaly-oriented decision support.

### Sequence Modeling Setup

To capture temporal behavior, the tabular grid signals are converted into fixed-length time windows before modeling.

**Window configuration**

- Sequence length: **12**
- Number of generated windows: **9,989**
- Window tensor shape: **(12, 5)**

The pipeline then splits the problem into two stages:

1. **LSTM autoencoder** trained only on normal windows for anomaly-sensitive reconstruction.
2. **LSTM classifier** trained on anomalous windows for downstream fault-type classification.

## Model Training

The notebook includes model training and evaluation for the sequence-learning stage.

![Autoencoder Training Curve](images/autoencoder_training_curve.png)

## Evaluation

Because this project involves classification across multiple grid conditions, confusion matrices are among the most informative result visuals for the README.

![Autoencoder Confusion Matrix](images/autoencoder_conf_matrix.png)

![Classifier Confusion Matrix](images/classifier_conf_matrix.png)

## Evaluation

### LSTM Autoencoder Results

The LSTM autoencoder is trained only on **pure normal** sequence windows to learn healthy operating behavior.

**Autoencoder training setup**

- Training normal windows: **1,671**
- Architecture: **LSTM(5 → 64) encoder + LSTM decoder + linear head (64 → 32 → 5)**
- Loss function: **MSELoss**
- Optimizer: **Adam**
- Epochs: **1000**

**Threshold selection**
A scan over reconstruction-error thresholds identified **4.5** as the best operating threshold by F1-score.

**Best anomaly detection threshold**

- Threshold: **4.5**
- F1-score: **0.9969**
- Anomaly recall: **0.9986**
- False positive rate: **0.0126**

**Autoencoder anomaly-vs-normal performance**

- Accuracy: **0.9955**
- Macro F1-score: **0.9944**
- Weighted F1-score: **0.9955**

**Binary classification report**

- Normal class: precision **0.9964**, recall **0.9874**, F1-score **0.9919**
- Anomaly class: precision **0.9952**, recall **0.9986**, F1-score **0.9969**

These results show that reconstruction error is highly effective for separating normal and anomalous grid behavior in this sequence setting.

### Downstream Fault Classifier

After anomaly-sensitive feature learning, the project trains an **LSTM classifier** to predict the anomaly category.

**Classifier configuration**

- Input dimension: **5**
- Hidden dimension: **64**
- LSTM layers: **2**
- Dropout: **0.2**
- Output classes: **4** (`Voltage`, `Frequency`, `Power Factor`, `Combined`)

During training, the classifier reached validation accuracy up to **99.93%**, showing strong separability among anomaly categories in the prepared sequence windows.

## Repository Structure

- `1.Power Grid Anomaly Detection (LSTM Autoencoder + Streamlit).ipynb` — main notebook for preprocessing, training, and evaluation.
- `grid_anomaly/` — deployment-oriented project files including app and model components.
- `requirements.txt` — dependency file for reproducibility.

## Applications

This project is relevant to:

- smart grid monitoring,
- anomaly-aware grid diagnostics,
- fault classification in electrical systems,
- AI-assisted condition monitoring for power networks.

## Future Improvements

- Add stronger repository cleanup and dependency specification for easier reproduction.
- Extend benchmarking with additional anomaly detection baselines.
- Improve deployment packaging for smoother end-user execution.

## Author

Abhishek Pawar
