import os
import datetime
import pandas as pd
import mlflow
from src.utils import create_dummy_cols
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import joblib
import json

# Constants used:
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = "./artifacts/train_data_gold.csv"
data_version = "00000"
experiment_name = current_date

# Create necessary directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)

mlflow.set_experiment(experiment_name)

# Load data
data = pd.read_csv(data_gold_path)

# Data type splitting for categorical variables, dropping unnecessary cols
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

# Dummy encoding for categorical variables
# Create one-hot encoded cols for cat vars
for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

# Change to floats
for col in data:
    data[col] = data[col].astype("float64")

# Splitting the data into features and target
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

# Train-test split with stratification to preserve the same distribution across sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"],
}

model_grid = RandomizedSearchCV(
    model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10
)

model_grid.fit(X_train, y_train)

# Get the best model parameters
best_model_xgboost_params = model_grid.best_params_

# Predictions
y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)

# Confusion matrix for test and train saved in /reports/figures
output_dir = "./reports/figures/"
os.makedirs(output_dir, exist_ok=True)

# Test set confusion matrix and report
conf_matrix = confusion_matrix(y_test, y_pred_test)
test_crosstab = pd.crosstab(
    y_test, y_pred_test, rownames=["Actual"], colnames=["Predicted"], margins=True
)

test_crosstab.to_csv(os.path.join(output_dir, "XGBoost_confusion_matrix_test.csv"))

test_report = classification_report(y_test, y_pred_test)
with open(os.path.join(output_dir, "XGBoost_classification_report_test.txt"), "w") as f:
    f.write(test_report)

# Train set confusion matrix and report
conf_matrix = confusion_matrix(y_train, y_pred_train)
train_crosstab = pd.crosstab(
    y_train, y_pred_train, rownames=["Actual"], colnames=["Predicted"], margins=True
)

train_crosstab.to_csv(os.path.join(output_dir, "XGBoost_confusion_matrix_train.csv"))

train_report = classification_report(y_train, y_pred_train)
with open(
    os.path.join(output_dir, "XGBoost_classification_report_train.txt"), "w"
) as f:
    f.write(train_report)

# Save the best XGBoost model
xgboost_model = model_grid.best_estimator_
xgboost_model_path = "./artifacts/lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}


# Logistic Regression model for comparison
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "./artifacts/lead_model_lr.pkl"

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01],
    }
    model_grid = RandomizedSearchCV(
        model, param_distributions=params, verbose=3, n_iter=10, cv=3
    )
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    # log artifacts
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")

    # store model for model interpretability
    joblib.dump(value=best_model, filename=lr_model_path)

    # Custom python model for predicting probability
    mlflow.pyfunc.log_model("model", python_model=lr_wrapper(best_model))


model_classification_report = classification_report(
    y_test, y_pred_test, output_dict=True
)

best_model_lr_params = model_grid.best_params_

# Confusion matrix for test and train saved in /reports/figures

# Test set confusion matrix and report
conf_matrix = confusion_matrix(y_test, y_pred_test)
test_crosstab = pd.crosstab(
    y_test, y_pred_test, rownames=["Actual"], colnames=["Predicted"], margins=True
)

test_crosstab.to_csv(os.path.join(output_dir, "LR_confusion_matrix_test.csv"))

test_report = classification_report(y_test, y_pred_test)
with open(os.path.join(output_dir, "LR_classification_report_test.txt"), "w") as f:
    f.write(test_report)

# Train set confusion matrix and report
conf_matrix = confusion_matrix(y_train, y_pred_train)
train_crosstab = pd.crosstab(
    y_train, y_pred_train, rownames=["Actual"], colnames=["Predicted"], margins=True
)

train_crosstab.to_csv(os.path.join(output_dir, "LR_confusion_matrix_train.csv"))

train_report = classification_report(y_train, y_pred_train)
with open(os.path.join(output_dir, "LR_classification_report_train.txt"), "w") as f:
    f.write(train_report)

# Save the best Logistic Regression model
model_results[lr_model_path] = model_classification_report

# Save column list and model results
column_list_path = "./artifacts/columns_list.json"
with open(column_list_path, "w+") as columns_file:
    columns = {"column_names": list(X_train.columns)}
    json.dump(columns, columns_file)

model_results_path = "./artifacts/model_results.json"
with open(model_results_path, "w+") as results_file:
    json.dump(model_results, results_file)
