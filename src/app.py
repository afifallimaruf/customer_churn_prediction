import streamlit as st
import pandas as pd
import mlflow.sklearn

def load_model(model_path):

    model = mlflow.sklearn.load_model(model_path + "/logistic_regression")
    return model, "sklearn"

st.title("Customer Churn Prediction")

# Input fields
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)


# Preprocess inputs
input_data = pd.DataFrame({
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    'tenure': [tenure],
    "PhoneService": 1 if phone_service == "Yes" else 0,
    "MultipleLines": {"Yes": 1, "No": 0, "No phone service": 2}[multiple_lines],
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service],
    "OnlineSecurity": {"Yes": 1, "No": 0, "No internet service": 2}[online_security],
    "OnlineBackup": {"Yes": 1, "No": 0, "No internet service": 2}[online_backup],
    "DeviceProtection": {"Yes": 1, "No": 0, "No internet service": 2}[device_protection],
    "TechSupport": {"Yes": 1, "No": 0, "No internet service": 2}[tech_support],
    "StreamingTV": {"Yes": 1, "No": 0, "No internet service": 2}[streaming_tv],
    "StreamingMovies": {"Yes": 1, "No": 0, "No internet service": 2}[streaming_movies],
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
    "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
    "PaymentMethod": {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    
    # Add other features as needed
})

# Load model
model, model_type = load_model("models/saved_models")

# Predict
if st.button("Predict"):
    if model_type == "sklearn":
        prediction = model.predict(input_data)[0]
    
    st.write(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")