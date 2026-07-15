import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# --------------------------------------------------
# Hugging Face Authentication
# --------------------------------------------------

token = os.getenv("HF_TOKEN")

if token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=token)

# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tourism-package-prediction")

# --------------------------------------------------
# Load Dataset from Hugging Face Dataset Repository
# --------------------------------------------------

repo_id = "PreethiKarunanithi/Preethi-first-project-space"

Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").squeeze()

print("Training data loaded successfully.")

# --------------------------------------------------
# Calculate Class Weight
# --------------------------------------------------

class_counts = ytrain.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]

# --------------------------------------------------
# Define XGBoost Model
# --------------------------------------------------

model = xgb.XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {
    "n_estimators": [50, 75],
    "max_depth": [2, 3],
    "learning_rate": [0.01, 0.05],
    "colsample_bytree": [0.4, 0.5],
    "colsample_bylevel": [0.4, 0.5],
    "reg_lambda": [0.4, 0.5]
}

# --------------------------------------------------
# Train Model
# --------------------------------------------------

with mlflow.start_run():

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)

    train_pred = best_model.predict(Xtrain)
    test_pred = best_model.predict(Xtest)

    train_report = classification_report(
        ytrain,
        train_pred,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        test_pred,
        output_dict=True
    )

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "test_accuracy": test_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "test_precision": test_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "test_recall": test_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_f1": test_report["1"]["f1-score"]
    })

    model_path = "best_tourism_package_model_v1.joblib"

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path)

    print("Model saved successfully.")

# --------------------------------------------------
# Upload Model to Hugging Face Model Hub
# --------------------------------------------------

model_repo = "PreethiKarunanithi/tourism-package-model"

try:
    api.repo_info(
        repo_id=model_repo,
        repo_type="model"
    )
    print("Model repository already exists.")

except RepositoryNotFoundError:
    print("Creating model repository...")

    create_repo(
        repo_id=model_repo,
        repo_type="model",
        private=False,
        token=token
    )

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=model_repo,
    repo_type="model"
)

print("Model uploaded successfully.")
