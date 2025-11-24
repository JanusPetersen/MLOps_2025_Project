import os
import datetime
import pandas as pd
import mlflow
from utils import create_dummy_cols
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
    "eval_metric": ["aucpr", "error"]
}

model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

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
test_crosstab = pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True)

test_crosstab.to_csv(os.path.join(output_dir, "confusion_matrix_test.csv"))

test_report = classification_report(y_test, y_pred_test)
with open(os.path.join(output_dir, "classification_report_test.txt"), "w") as f:
    f.write(test_report)

# Train set confusion matrix and report
conf_matrix = confusion_matrix(y_train, y_pred_train)
train_crosstab = pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True)

train_crosstab.to_csv(os.path.join(output_dir, "confusion_matrix_train.csv"))

train_report = classification_report(y_train, y_pred_train)
with open(os.path.join(output_dir, "classification_report_train.txt"), "w") as f:
    f.write(train_report)