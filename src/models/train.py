import os
import datetime
import pandas as pd
import mlflow
from utils import create_dummy_cols

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