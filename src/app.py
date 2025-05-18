import streamlit as st
import pandas as pd
import mlflow.sklearn
import mlflow.pytorch
import torch

def load_model(model_path):
    try:
        model = mlflow.sklearn.load_model(model_path + "/logistic_regression")
        return model, "sklearn"
    except:
        model = mlflow.pytorch.load_model(model_path + "/neural_network")
        return model, "pytorch"

st.title("Customer Churn Prediction")

# Input fields
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)
gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Preprocess inputs
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'gender': [1 if gender == "Male" else 0],
    'Contract': [0 if contract == "Month-to-month" else 1 if contract == "One year" else 2]
    # Add other features as needed
})

# Load model
model, model_type = load_model("models/saved_models")

# Predict
if st.button("Predict"):
    if model_type == "sklearn":
        prediction = model.predict(input_data)[0]
    else:
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = (model(input_tensor) > 0.5).float().item()
    
    st.write(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")