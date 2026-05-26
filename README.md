# Power Grid Anomaly Detection

Deep learning-based power grid anomaly detection using sequence modeling, reconstruction error analysis, and classification for grid event monitoring.

![Application Interface](application_interface.png)

## Overview

This project presents a deep learning pipeline for detecting anomalies in power grid data using sequence-based modeling. It combines an LSTM autoencoder for anomaly-sensitive feature learning with a downstream classifier for fault category prediction, and also includes a Streamlit interface for interactive analysis and visualization.[page:0][page:1]

## Key Features

- LSTM-based sequence modeling for power grid anomaly detection.[page:0][page:1]
- Autoencoder-driven latent feature learning from multivariate grid signals.[page:0][page:1]
- Multi-class classification of grid conditions including normal and anomalous states.[page:0]
- Streamlit application for interactive fault analysis and visualization.[page:0]
- Modular project structure with saved model weights and deployment files.[page:1]

## Dataset and Classes

The project works on multivariate power system measurements and frames anomaly detection as a multi-class learning problem.[page:0]  
The notebook defines the following operating classes:[page:0]

- Normal
- Voltage Anomaly
- Frequency Anomaly
- Power Factor Anomaly
- Combined Anomaly

![Anomaly Reference](anomaly_reference.png)

![Class Distribution](class_distribution.png)

## Methodology

The workflow begins with preprocessing and organizing the power grid signals into sequence windows suitable for recurrent modeling.[page:0]  
An LSTM autoencoder is used to learn sequence representations, after which learned features are used for downstream classification of grid conditions.[page:0][page:1]  
This design allows the project to combine temporal pattern learning with anomaly-oriented decision support.[page:0][page:1]

## Model Training

The notebook includes model training and evaluation for the sequence-learning stage, making the training behavior itself useful to show in the README.[page:0]  
A training-curve figure helps demonstrate convergence and gives visitors quick evidence that the model was actually trained and monitored rather than only implemented.[page:0]

![Autoencoder Training Curve](autoencoder_training_curve.png)

## Evaluation

Because this project involves classification across multiple grid conditions, confusion matrices are among the most informative result visuals for the README.[page:0]  
They show where the model distinguishes well between classes and where overlap remains between anomaly categories.[page:0]

![Autoencoder Confusion Matrix](autoencoder_conf_matrix.png)

![Classifier Confusion Matrix](classifier_conf_matrix.png)

## Repository Structure

- `1.Power Grid Anomaly Detection (LSTM Autoencoder + Streamlit).ipynb` — main notebook for preprocessing, training, and evaluation.[page:0]
- `grid_anomaly/` — deployment-oriented project files including app and model components.[page:1]
- `requirements.txt` — dependency file for reproducibility.[page:2]

## Applications

This project is relevant to:

- smart grid monitoring,
- anomaly-aware grid diagnostics,
- fault classification in electrical systems,
- AI-assisted condition monitoring for power networks.[page:0][page:1]

## Future Improvements

- Add stronger repository cleanup and dependency specification for easier reproduction.[page:1][page:2]
- Extend benchmarking with additional anomaly detection baselines.[cite:261]
- Improve deployment packaging for smoother end-user execution.[page:1]

## Author

Abhishek Pawar