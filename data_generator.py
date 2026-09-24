import pandas as pd
import numpy as np

def generate_hospital_data(hospital_name, num_records, profile):
    np.random.seed(hash(hospital_name) % (2**32))
    
    # Base features
    patient_ids = [f"{hospital_name}_{i}" for i in range(num_records)]
    
    if profile == 'A':
        # Older, cardiac/hypertension, high readmission
        age = np.random.normal(70, 10, num_records)
        systolic_bp = np.random.normal(150, 20, num_records)
        diastolic_bp = np.random.normal(90, 15, num_records)
        comorbidity_index = np.random.poisson(3, num_records)
        readmission_base_prob = 0.35
    elif profile == 'B':
        # Moderate age, respiratory, smoking, moderate readmission
        age = np.random.normal(60, 12, num_records)
        systolic_bp = np.random.normal(130, 15, num_records)
        diastolic_bp = np.random.normal(80, 10, num_records)
        comorbidity_index = np.random.poisson(2, num_records)
        readmission_base_prob = 0.25
    else:
        # Younger, diabetes, low readmission
        age = np.random.normal(45, 15, num_records)
        systolic_bp = np.random.normal(120, 10, num_records)
        diastolic_bp = np.random.normal(75, 8, num_records)
        comorbidity_index = np.random.poisson(1, num_records)
        readmission_base_prob = 0.15
        
    age = np.clip(age, 18, 95).astype(int)
    
    # Shared random features
    gender = np.random.choice(['Male', 'Female'], num_records)
    blood_group = np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], num_records)
    admission_type = np.random.choice(['Emergency', 'Elective', 'Transfer'], num_records)
    department = np.random.choice(['Cardiology', 'Pulmonology', 'Endocrinology', 'General'], num_records)
    
    heart_rate = np.random.normal(80, 15, num_records).astype(int)
    respiratory_rate = np.random.normal(18, 4, num_records).astype(int)
    temperature = np.random.normal(37, 0.5, num_records)
    spo2 = np.random.normal(95, 3, num_records)
    glucose = np.random.normal(100 if profile != 'C' else 140, 20 if profile != 'C' else 40, num_records)
    hba1c = np.random.normal(5.5 if profile != 'C' else 7.0, 0.5 if profile != 'C' else 1.5, num_records)
    wbc_count = np.random.normal(8.0, 2.0, num_records)
    hemoglobin = np.random.normal(14.0, 1.5, num_records)
    platelet_count = np.random.normal(250, 50, num_records).astype(int)
    creatinine = np.random.normal(1.0, 0.2, num_records)
    alt = np.random.normal(25, 10, num_records).astype(int)
    ast = np.random.normal(25, 10, num_records).astype(int)
    bmi = np.random.normal(28, 5, num_records)
    
    smoking_history = np.random.choice(['Never', 'Former', 'Current'], num_records, p=[0.5, 0.3, 0.2] if profile != 'B' else [0.2, 0.4, 0.4])
    alcohol_use = np.random.choice(['None', 'Moderate', 'Heavy'], num_records)
    diagnosis = np.random.choice(['D1', 'D2', 'D3'], num_records)
    length_of_stay = np.random.poisson(5, num_records)
    treatment_cost = np.random.normal(5000, 1500, num_records)
    
    # Calculate target (Readmission / Mortality)
    # Target is Outcome (0 = Discharged, 1 = Adverse Outcome)
    
    risk_score = readmission_base_prob + (age > 70) * 0.1 + (comorbidity_index > 2) * 0.15 + (spo2 < 92) * 0.1 + (systolic_bp > 140) * 0.05
    probabilities = np.clip(risk_score + np.random.normal(0, 0.1, num_records), 0, 1)
    outcome = np.random.binomial(1, probabilities)
    
    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': age,
        'Gender': gender,
        'Blood_Group': blood_group,
        'Admission_Type': admission_type,
        'Department': department,
        'Heart_Rate': heart_rate,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'Respiratory_Rate': respiratory_rate,
        'Temperature': temperature,
        'SpO2': spo2,
        'Glucose': glucose,
        'HbA1c': hba1c,
        'WBC_Count': wbc_count,
        'Hemoglobin': hemoglobin,
        'Platelet_Count': platelet_count,
        'Creatinine': creatinine,
        'ALT': alt,
        'AST': ast,
        'BMI': bmi,
        'Smoking_History': smoking_history,
        'Alcohol_Use': alcohol_use,
        'Comorbidity_Index': comorbidity_index,
        'Diagnosis': diagnosis,
        'Length_of_Stay': length_of_stay,
        'Treatment_Cost': treatment_cost,
        'Outcome': outcome
    })
    
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    df_a = generate_hospital_data('HOSPITAL_A', 5000, 'A')
    df_a.to_csv('data/HOSPITAL_A.csv', index=False)
    print("Generated HOSPITAL_A.csv (Profile A: More cardiac, higher age)")
    
    df_b = generate_hospital_data('HOSPITAL_B', 5000, 'B')
    df_b.to_csv('data/HOSPITAL_B.csv', index=False)
    print("Generated HOSPITAL_B.csv (Profile B: More respiratory, smoking)")
    
    df_c = generate_hospital_data('HOSPITAL_C', 5000, 'C')
    df_c.to_csv('data/HOSPITAL_C.csv', index=False)
    print("Generated HOSPITAL_C.csv (Profile C: More diabetes, younger)")
    
    print("\nData generation complete.")
