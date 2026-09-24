import argparse
import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# Define PyTorch Model
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_data(hospital_id):
    df = pd.read_csv(f'data/HOSPITAL_{hospital_id}.csv')
    
    # Preprocessing
    df = df.drop(columns=['Patient_ID', 'Diagnosis'])
    
    # Categorical encoding
    categorical_cols = ['Gender', 'Blood_Group', 'Admission_Type', 'Department', 'Smoking_History', 'Alcohol_Use']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save the scaler and encoders so the dashboard can use them for inference
    os.makedirs('models', exist_ok=True)
    if hospital_id == 'A':  # Just save one global scaler representing the expected feature format
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoders, 'models/encoders.pkl')
        joblib.dump(df.drop(columns=['Outcome']).columns.tolist(), 'models/features.pkl')
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, X.shape[1]

def automl_search(model, train_loader, test_loader):
    print("Running Local AutoML Hyperparameter Search...")
    best_lr = 0.001
    best_loss = float('inf')
    
    for lr in [0.01, 0.001, 0.0001]:
        # Quick eval
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        
        # train for 1 epoch to check performance
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            break # Just 1 batch for fast search in this simulation
            
        test_loss, _, _, _ = test(model, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            best_lr = lr
            
    print(f"AutoML Search Complete. Best LR: {best_lr}")
    return best_lr

def train(model, train_loader, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss += criterion(output.squeeze(), target).item()
            probs = torch.sigmoid(output).squeeze()
            preds = (probs > 0.5).float()
            
            all_targets.extend(target.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            
    loss /= len(test_loader)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
        
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return loss, accuracy, auc, f1

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, best_lr):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_lr = best_lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=5, lr=self.best_lr)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, auc, f1 = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy), "auc": float(auc), "f1": float(f1)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--hospital-id", choices=['A', 'B', 'C'], required=True, type=str, help="Hospital Profile ID (A, B, or C)")
    args = parser.parse_args()

    train_loader, test_loader, input_dim = load_data(args.hospital_id)
    model = Net(input_dim)
    
    best_lr = automl_search(model, train_loader, test_loader)
    
    print(f"Starting Flower Client for Hospital {args.hospital_id}...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=HospitalClient(model, train_loader, test_loader, best_lr),
    )
