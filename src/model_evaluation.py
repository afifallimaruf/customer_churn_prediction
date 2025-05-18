import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import f1_score
import wandb
import os


def evaluate_models(data_path, model_path):
    # initialize weights & biases
    # wandb.init(project="customer-churn", mode="online")

    # load data
    df = pd.read_csv(data_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    logistic_regression = os.path.join("models/saved_models", "logistic_regression")
    neural_network = os.path.join("models/saved_models", "neural_network")
    random_forest = os.path.join("models/saved_models", "random_forest") 
    
    # if logistic_regression:
    #     # start mlflow run
    #     with mlflow.start_run():
    #         # load and evaluate logistic regression
    #         lr = mlflow.sklearn.load_model(model_path + "/logistic_regression")
    #         lr_pred = lr.predict(X)
    #         lr_f1 = f1_score(y, lr_pred)
    #         mlflow.log_metric("lr_f1_score", lr_f1)
    #         # wandb.log({"lr_f1_score": lr_f1})
    # elif neural_network:
    #     # load and evaluate neural network
    #     nn = mlflow.pytorch.load_model(model_path + "/neural_network")
    #     X_tensor = torch.tensor(X.values, dtype=torch.float32)
    #     nn.eval()
    #     with torch.no_grad():
    #         nn_pred = nn(X_tensor)
    #         nn_pred = (nn_pred > 0.5).float()
    #         nn_f1 = f1_score(y, nn_pred)
    #     mlflow.log_metric("nn_f1_score", nn_f1)
    #     # wandb.log({"nn_f1_score": nn_f1})
    # elif random_forest:
    with mlflow.start_run():
        # load and evaluate logistic regression
        rf = mlflow.sklearn.load_model(model_path + "/random_forest")
        rf_pred = rf.predict(X)
        rf_f1 = f1_score(y, rf_pred)
        mlflow.log_metric("rf_f1_score", rf_f1)
        # wandb.log({"lr_f1_score": lr_f1})

if __name__ == "__main__":
    evaluate_models('dataset/processed/processed_data.csv', 'models/saved_models')