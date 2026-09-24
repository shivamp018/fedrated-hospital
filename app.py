import streamlit as st
import numpy as np
import torch
import pandas as pd
from client import Net
import time
import joblib
import json
import os

st.set_page_config(page_title="Federated AutoML in Hospital", layout="wide", page_icon="🏥")

st.title("🏥 Auto ML in Hospital using Federated Architecture")
st.subheader("Predicting Readmission Risk & In-hospital Mortality")

st.markdown("""
### 🔒 Privacy-Preserving | Collaborative | Scalable | Secure

This application demonstrates the **Inference Phase** of our Federated Learning Architecture. 
Crucially, **where does the prediction take place?**
""")

st.info("""
**Prediction takes place LOCALLY at the Hospital (Client).**
When you enter patient data below, it is **never sent to the central server**. 
Instead, the hospital downloads the **Aggregated Global Model Weights** from the central server and runs the prediction locally on their own infrastructure. 
This ensures complete privacy of sensitive patient records!
""")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Local Patient Data Input")
    st.markdown("*(Data remains on hospital premises)*")
    
    age = st.slider("Age", 18, 95, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    systolic_bp = st.number_input("Systolic BP", 80, 200, 145)
    diastolic_bp = st.number_input("Diastolic BP", 40, 120, 90)
    heart_rate = st.number_input("Heart Rate", 40, 150, 85)
    spo2 = st.number_input("SpO2 (%)", 80, 100, 94)
    comorbidity = st.slider("Comorbidity Index", 0, 10, 3)
    
    predict_btn = st.button("Run Local Prediction", use_container_width=True, type="primary")

with col2:
    st.header("2. Federated Architecture Workflow")
    
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        if metrics:
            st.markdown("### 📈 Global Model Training Performance (20 Rounds)")
            df_metrics = pd.DataFrame(metrics)
            st.line_chart(df_metrics.set_index("round")[["accuracy", "auc"]])
    
    if not predict_btn:
        st.markdown("""
        **How the Model Was Trained:**
        1. **Hospital A, B, and C** generated local data (Non-IID).
        2. Each hospital ran **AutoML Search** to find optimal learning rates.
        3. Only **model weights** (not patient data) were sent to the **Central Server**.
        4. The server performed **Secure Aggregation (FedAvg)** over 20 rounds.
        5. The resulting **Global Model** was sent back to all hospitals.
        
        *Awaiting local prediction...*
        """)
    else:
        st.markdown("### 🔄 Executing Local Inference Pipeline...")
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.text("1. Loading Aggregated Global Model Weights from Server...")
        progress_bar.progress(25)
        time.sleep(1)
        
        try:
            # Load Global Model
            data = np.load("global_model_weights.npz")
            weights = [data[key] for key in data.files]
            input_dim = weights[0].shape[1]
            
            model = Net(input_dim)
            params_dict = zip(model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            status_text.text("2. Global Model successfully deployed to Local Hospital Node!")
            progress_bar.progress(50)
            time.sleep(1)
            
            status_text.text("3. Processing patient data locally (Privacy Preserved)...")
            
            # Load Scaler and Encoders
            scaler = joblib.load('models/scaler.pkl')
            encoders = joblib.load('models/encoders.pkl')
            features = joblib.load('models/features.pkl')
            
            # Create a dict with average/default values for all features
            patient_data = {
                'Age': age,
                'Gender': gender,
                'Blood_Group': 'O+',
                'Admission_Type': 'Emergency',
                'Department': 'Cardiology',
                'Heart_Rate': heart_rate,
                'Systolic_BP': systolic_bp,
                'Diastolic_BP': diastolic_bp,
                'Respiratory_Rate': 18,
                'Temperature': 37.0,
                'SpO2': spo2,
                'Glucose': 100,
                'HbA1c': 5.5,
                'WBC_Count': 8.0,
                'Hemoglobin': 14.0,
                'Platelet_Count': 250,
                'Creatinine': 1.0,
                'ALT': 25,
                'AST': 25,
                'BMI': 28.0,
                'Smoking_History': 'Never',
                'Alcohol_Use': 'None',
                'Comorbidity_Index': comorbidity,
                'Length_of_Stay': 5,
                'Treatment_Cost': 5000.0
            }
            
            df_input = pd.DataFrame([patient_data])
            # Ensure columns are in the exact same order as training
            df_input = df_input[features]
            
            # Apply Encoders
            for col in encoders:
                # Handle unseen labels gracefully just in case
                known_classes = encoders[col].classes_
                val = df_input[col].iloc[0]
                if val not in known_classes:
                    df_input[col] = 0 # fallback
                else:
                    df_input[col] = encoders[col].transform(df_input[col])
                    
            # Apply Scaler
            X_input = scaler.transform(df_input.values)
            input_tensor = torch.tensor(X_input, dtype=torch.float32)
            
            progress_bar.progress(75)
            time.sleep(1)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
            
            status_text.text("4. Prediction Complete!")
            progress_bar.progress(100)
            
            st.divider()
            st.subheader("📊 Local Prediction Results")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Readmission / Mortality Risk", f"{prob*100:.1f}%")
            
            if prob > 0.5:
                metric_col2.error("⚠️ High Risk - Clinical Decision Support Alerted")
            else:
                metric_col2.success("✅ Low Risk - Safe for Discharge")
                
            st.caption("Notice how no patient features (Age, BP, etc.) were transmitted during this process!")
            
        except Exception as e:
            st.error(f"Failed to load global weights or process data. Did you run the federated training? Error: {e}")
