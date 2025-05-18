import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import yaml
import wandb
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


def train_models(data_path, model_path):
    # initialize weights & biases
    # wandb.init(project="customer-churn", config=params)

    # load data
    df = pd.read_csv(data_path)
    
    # split data to features & target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # split data into train & test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="random_forest_classifier"):
        print("Training Random Forest model start...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # log metrics for random forest classifier
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_precision = precision_score(y_test, rf_pred)
        rf_recall = recall_score(y_test, rf_pred)
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_metric("accuracy", rf_accuracy)
        mlflow.log_metric("precision", rf_precision)
        mlflow.log_metric("recall", rf_recall)
        mlflow.sklearn.log_model(rf, "random_forest")
        print("Training Random Forest model finish...")
        print(f'Accuracy: {rf_accuracy}')

    # start mlflow run
    with mlflow.start_run(run_name="logistic_regression"):
        print("Training Logistic Regression model start...")
        # train logistic regression
        lr = LogisticRegression(max_iter=params['lr_max_iter'])
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # log metrics for logistic regression
        lr_accuracy = accuracy_score(y_test, lr_pred)
        lr_precision = precision_score(y_test, lr_pred)
        lr_recall = recall_score(y_test, lr_pred)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("lr_max_iter", params['lr_max_iter'])
        mlflow.log_metric("accuracy", lr_accuracy)
        mlflow.log_metric("precision", lr_precision)
        mlflow.log_metric("recall", lr_recall)
        mlflow.sklearn.log_model(lr, "logistic_regression")
        print("Training Logistic Regression model finish...")
        print(f'Accuracy: {lr_accuracy}')
        # wandb.log({"lr_accuracy": lr_accuracy, "lr_precision": lr_precision, "lr_recall": lr_recall})

        # save best model (based on accuracy)
        # best_model = lr if lr_accuracy > nn_accuracy else model
        os.makedirs(model_path, exist_ok=True)
        if lr_accuracy > rf_accuracy:
            mlflow.sklearn.save_model(lr, model_path + "/logistic_regression")
        else:
            mlflow.sklearn.save_model(rf, model_path + "/random_forest")

if __name__ == "__main__":
    train_models("dataset/processed/processed_data.csv", "models/saved_models")