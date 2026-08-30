# Headcount Attrition - Survival Analysis

**CV skill represented:** Survival & Reliability Analysis (Cox Proportional Hazards), Applied Statistics (log-rank test), classification model comparison & tuning.

## Problem
Two related HR questions on the public IBM HR Analytics dataset:
1. **Who** is likely to resign? — classification (Logistic Regression / Random Forest / Gradient Boosting, compared and tuned).
2. **When** are they likely to resign? — a Cox Proportional Hazards survival model estimating each employee's hazard of leaving over time.

## Method
- 5-fold stratified CV model comparison, then randomized hyperparameter search on the best model, evaluated on a held-out test set (ROC-AUC, PR-AUC, recall).
- Cox PH model on tenure as time-to-event, with Kaplan-Meier retention curves and a log-rank test (overtime vs. no-overtime).
- Results are assembled into an auto-generated executive PowerPoint report, with remarks generated dynamically from the fitted models rather than hardcoded.

## Result
Best classifier: ROC-AUC ≈ 0.81 (test). Cox model: concordance index ≈ 0.85 (test).

## How to run
Open `code/hr_attrition_analysis.ipynb` and run top to bottom. Data: `data/IBM-HR-Employee-Attrition.csv`. Outputs (figures + `Executive_Attrition_Report.pptx`) are written to `result/`.
