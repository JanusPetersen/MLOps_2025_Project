# MLOps Lead Prediction Pipeline

A containerized machine learning pipeline for predicting lead conversion using Dagger, Docker, and GitHub Actions. This project demonstrates modern MLOps practices including automated CI/CD, model versioning with MLflow, and data version control with DVC.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Getting Started](#getting-started)
- [GitHub Actions Workflow](#github-actions-workflow)
- [Local Development](#local-development)
- [Model Artifacts](#model-artifacts)
- [Data Management](#data-management)
- [CI/CD Pipeline Flow](#ci/cd-pipeline-flow)

## Project Overview

This project implements an end-to-end machine learning pipeline that:
- Processes raw lead data with automated preprocessing
- Trains multiple models (XGBoost, Logistic Regression) using grid search
- Evaluates model performance with MLflow tracking
- Automatically deploys the best performing model
- Validates trained models using automated inference tests
- Runs entirely in containerized environments for reproducibility

**Key Features:**
- Fully containerized pipeline using Dagger
- Automated CI/CD with GitHub Actions
- Data versioning with DVC
- Model tracking and registry with MLflow
- Automated model validation
- Reproducible builds with Docker

## Architecture

The pipeline follows a modular architecture with clear separation of concerns:

```mermaid
flowchart TD
    A[GitHub Actions Runner] --> B[Checkout Code]
    B --> C[Pull Data via DVC<br/>raw_data.csv]
    C --> D[Run Dagger Pipeline]
    
    D --> E
    
    subgraph DAGGER["Dagger Container (Python 3.10)"]
        E[Data Processing]
        F[Train Models]
        G[Evaluate & Select]
        H[Deploy Best Model]
        I[artifacts/]
        
        E -->|train_data_gold.csv| F
        F -->|models/| G
        G -->|mlruns/| H
        H -->|register| I
    end
    
    I --> J[Export Artifacts]
    J --> K[Upload model.pkl]
    K --> L[Model Inference Test]
    
    style DAGGER fill:none,stroke:#333,stroke-width:2px
    style L fill:#4CAF50,stroke:#333,color:#000
```


## Project Structure

```
MLOps_2025_Project/
├── .github/
│   └── workflows/
│       └── dagger.yml              # GitHub Actions CI/CD workflow
├── src/
│   ├── data/
│   │   ├── data.py                 # Data preprocessing pipeline
│   │   ├── raw_data.csv            # Raw input data (pulled via DVC)
│   │   └── raw_data.csv.dvc        # DVC pointer file
│   ├── models/
│   │   ├── train.py                # Model training (XGBoost, LogReg)
│   │   ├── evaluate.py             # Model evaluation & selection
│   │   └── Deploy.py               # Model registration & deployment
│   └── utils.py                    # Shared utility functions
├── workflow/
│   ├── pipeline.go                 # Dagger pipeline orchestration (Go)
│   └── go.mod                      # Go dependencies
├── artifacts/                      # Pipeline outputs (exported)
│   ├── train_data_gold.csv         # Processed training data
│   ├── lead_model_lr.pkl           # Trained Logistic Regression model
│   ├── lead_model_xgboost.json     # Trained XGBoost model
│   ├── training_data.csv           # Intermediate processed data
│   ├── X_test.csv, y_test.csv      # Test split data
│   └── *.json, *.csv               # Various metadata & reports
├── reports/
│   └── figures/                    # Confusion matrices & reports
├── Dockerfile                      # Container image definition
├── requirements.txt                # Python dependencies
├── .dvc/
│   ├── config                      # DVC configuration
│   └── cache/                      # DVC cached data files
└── README.md                       # This file
```

## Pipeline Stages

### 1. Data Processing (`src/data/data.py`)
**Purpose:** Clean, transform, and prepare raw lead data for training

**Operations:**
- Load raw CSV data
- Filter records by date range
- Handle missing values using imputation strategies
- Detect and handle outliers using IQR method
- Feature engineering: one-hot encoding for categorical variables
- Feature scaling: MinMax normalization
- Train/test split with stratification

**Outputs:**
- `train_data_gold.csv` - Final processed training dataset
- `training_data.csv` - Intermediate processed data
- `X_test.csv`, `y_test.csv` - Test split
- Metadata files: `date_limits.json`, `columns_drift.json`, `outlier_summary.csv`

### 2. Model Training (`src/models/train.py`)
**Purpose:** Train and optimize multiple ML models

**Models Trained:**
1. **XGBoost Random Forest Classifier**
   - Hyperparameter tuning via RandomizedSearchCV
   - Parameters: learning_rate, max_depth, subsample, etc.
   - 10-fold cross-validation
   
2. **Logistic Regression**
   - Hyperparameter tuning: solver, penalty, regularization (C)
   - 3-fold cross-validation
   - Saved as `lead_model_lr.pkl` for inference

**MLflow Integration:**
- Experiment tracking with timestamps
- Metrics logged: F1-score, accuracy, confusion matrix
- Model artifacts logged
- Parameters tracked

**Outputs:**
- `lead_model_xgboost.json` - XGBoost model
- `lead_model_lr.pkl` - Logistic Regression model
- `model_results.json` - Performance metrics
- Confusion matrices and classification reports in `reports/figures/`

### 3. Model Evaluation (`src/models/evaluate.py`)
**Purpose:** Compare trained models and select the best performer

**Process:**
- Query MLflow for experiments by date
- Compare F1-scores across models
- Select best model based on weighted F1-score
- Compare against production model
- Determine if new model should be promoted

**Outputs:**
- Model selection decision
- Performance comparison metrics

### 4. Model Deployment (`src/models/Deploy.py`)
**Purpose:** Register best model to MLflow Model Registry

**Operations:**
- Register selected model to MLflow
- Set model stage
- Track model versions

## Getting Started

### Prerequisites
- Docker Desktop installed and running
- Go 1.25.4 or later
- Python 3.10+
- Git with DVC installed

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JanusPetersen/MLOps_2025_Project.git
cd MLOps_2025_Project
```

2. **Pull data using DVC:**
```bash
cd src/data
dvc pull
cd ../..
```

3. **Install Python dependencies (optional, for local dev):**
```bash
pip install -r requirements.txt
```

## GitHub Actions Workflow

The CI/CD pipeline is defined in `.github/workflows/dagger.yml` and runs automatically on push to `main` branch.

### Workflow Steps

1. **Checkout Code** - Clone the repository
2. **Setup Go** - Install Go 1.25.4 for Dagger
3. **Install DVC** - Install DVC with all extras
4. **Pull Data** - Fetch `raw_data.csv` from DVC cache
5. **Run Dagger Pipeline** - Execute containerized ML pipeline
6. **Upload Artifacts** - Save all pipeline outputs
7. **Prepare Model Artifact** - Copy and rename model to `model/model.pkl`
8. **Upload Model** - Upload model artifact for validation
9. **Run Inference Test** - Validate model using external action

### Viewing Results

After a workflow run:
1. Go to **Actions** tab in GitHub
2. Click on the latest workflow run
3. Download artifacts:
   - `pipeline-artifacts` - All pipeline outputs
   - `model` - Trained model file

### Manual Trigger

You can manually trigger the workflow:
```bash
# Push to main branch, or
# Use GitHub UI: Actions → Run Dagger pipeline → Run workflow
```

## Local Development

### Running the Pipeline Locally

```bash
cd workflow
go run .
```

This will:
- Build the Docker image from the Dockerfile
- Run all pipeline stages in the container
- Export artifacts to `../artifacts/` directory


## Model Artifacts

### Model Format
- **File:** `model.pkl`
- **Type:** Scikit-learn Logistic Regression
- **Size:** ~5-10 KB
- **Features:** 50+ features after one-hot encoding

### Model Performance Metrics
Models are evaluated on:
- **F1-Score (weighted)** - Primary metric
- **Accuracy**
- **Precision, Recall** (per class)
- **Confusion Matrix**

## Data Management

### Data Versioning with DVC

The project uses DVC to version control the raw dataset:

**DVC Configuration:**
- Raw data URL: `https://raw.githubusercontent.com/Jeppe-T-K/itu-sdse-project-data/refs/heads/main/raw_data.csv`
- DVC pointer: `src/data/raw_data.csv.dvc`
- Cache location: `.dvc/cache/`

**Data Pipeline:**
```
GitHub Raw URL → DVC Cache → src/data/raw_data.csv → Processing → artifacts/
```

## CI/CD Pipeline Flow

```mermaid
flowchart TD
    A[Push to main] --> B[GitHub Actions Triggered]
    
    B --> C[Setup Environment<br/>Go · DVC]
    C --> D[Pull Data from DVC]
    D --> E[Run Dagger Pipeline]
    
    E --> F
    
    subgraph DAGGER["Dagger Container - Docker"]
        F[Data Processing]
        G[Model Training]
        H[Model Evaluation]
        I[Model Deployment]
        J[artifacts/]
        
        F -->|processed data| G
        G -->|trained models| H
        H -->|best model| I
        I -->|export| J
    end
    
    J --> K[Export Artifacts]
    K --> L[Upload Model to GitHub]
    L --> M[Model Inference Validation<br/>External Action]
    
    subgraph VALIDATION["Inference Validation"]
        N[Download Model Artifact]
        O[Load Model<br/>scikit-learn]
        P[Run Inference Tests]
        Q[Validate Predictions]
        
        N --> O --> P --> Q
    end
    
    M --> N
    
    style DAGGER fill:none,stroke:#333,stroke-width:2px
    style VALIDATION fill:none,stroke:#333,stroke-width:2px
    style Q fill:#4CAF50,stroke:#333,color:#000
```

## Authors

- **Frederik Holbech** - faho@itu.dk
- **Janus Petersen**  - jspe@itu.dk

## Acknowledgments

- ITU BDS MLOPS'25 Course Staff
- [Dagger Documentation](https://docs.dagger.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Model Validator Action](https://github.com/lasselundstenjensen/itu-sdse-project-model-validator)
