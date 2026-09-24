# Federated AutoML Hospital Architecture 🏥

This project implements a privacy-preserving Federated Learning system across multiple hospitals to predict Readmission Risk and In-hospital Mortality. It uses **Flower (flwr)** for federated aggregation, **PyTorch** for local training, and **Streamlit** for the deployment dashboard.

## Overview
1. **Local AutoML Training**: 3 Hospital clients generate local data (Non-IID distributions) and search for optimal hyperparameters locally.
2. **Federated Aggregation**: A central server aggregates the model weights securely using `FedAvg` over 20 rounds without ever seeing the raw patient data.
3. **Inference Dashboard**: A Streamlit frontend allows doctors to run inferences locally using the downloaded global model.

## Setup & Training
To train the global model yourself:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate synthetic hospital data:**
   ```bash
   python data_generator.py
   ```
3. **Run Federated Training (Starts Server + 3 Clients):**
   ```bash
   bash run.sh
   ```
   *This will save `global_model_weights.npz`, `metrics.json`, and preprocessing scalers into the `models/` folder.*

## Deployment (Streamlit)
To launch the interactive dashboard (which uses the pre-trained global weights):
```bash
streamlit run app.py
```

This repository is pre-configured to be deployed easily on Streamlit Community Cloud. 
