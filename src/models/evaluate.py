import datetime
import mlflow
from utils import wait_until_ready
import json
import pandas as pd

# Constants used
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
artifact_path = "model"
model_name = "lead_model"
experiment_name = current_date

# Getting experiment model results
experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

# Getting best model based on f1_score
experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]

# Getting results from the newly trained model for evaluation against the current best model
with open("./artifacts/model_results.json", "r") as f:
    model_results = json.load(f)
results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

# Selecting the best model based on f1-score
best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name