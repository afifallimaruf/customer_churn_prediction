Customer Churn Prediction

This project builds a customer churn prediction model using scikit-learn and PyTorch, with automated training, evaluation, and deployment using CI/CD pipelines.

Setup

Clone the repository:

git clone <repository_url>
cd customer-churn-prediction

Install dependencies:

pip install -r requirements.txt

Initialize DVC and pull data:

dvc init
dvc pull

Set up MLflow and Weights & Biases:

Configure MLflow tracking URI.

Set up Weights & Biases API key.

Run the pipeline:

dvc repro

Deploy the Streamlit app:

streamlit run src/app.py

CI/CD

The pipeline is automated using GitHub Actions. It triggers on code or data changes, running preprocessing, training, evaluation, and deployment.

Dataset

The Telco Customer Churn dataset is used, available on Kaggle.
