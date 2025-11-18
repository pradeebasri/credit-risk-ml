# credit-risk-ml
# Interpretable Machine Learning: SHAP Analysis for Credit Risk Prediction

## Overview

This project builds and interprets a credit risk prediction model using the LendingClub-like dataset.  
Key focus: actionable explanations of predictions with **SHAP** values, aiding both compliance and fair loan decisions.

## Steps

1. **Data Preprocessing**
    - Handle missing values
    - Encode categorical `purpose`
    - Feature engineering (e.g., squared dti)

2. **Modeling**
    - Train XGBoost on SMOTE-balanced data (see `loan_data.csv`)
    - Key metrics: AUC-ROC, AUC-PR, F1

3. **Interpretability**
    - Global SHAP summary (png figures)
    - Top features analyzed and explained
    - Individual SHAP force plots: 3 borrower profiles

4. **Deliverables**
    - `loan_data.csv`
    - Python script/notebook (`.py`/`.ipynb`)
    - README.md
    - Model performance & SHAP summary report (see code output/logs)
    - PNGs of all SHAP plots

## How To Run

1. Clone/download files and ensure these packages: pandas, numpy, xgboost, imblearn, shap, matplotlib, seaborn, scikit-learn.
2. Run the provided script:  
    `python credit_shap_project.py`
3. Check generated figures in the script directory for all SHAP visualizations.

## Interpretation

- **High/Low risk drivers:** as outlined in the textual deliverables, based on SHAP.
- **Actionable use:** explanations guide both automated and manual credit review, supporting compliance.


