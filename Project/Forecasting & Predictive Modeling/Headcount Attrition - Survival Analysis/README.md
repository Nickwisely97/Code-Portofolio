# Headcount Attrition - Survival Analysis

**CV skill represented:** Survival & Reliability Analysis (Cox Proportional Hazards), Applied Statistics (log-rank test, proportional-hazards assumption check).

## Problem
One model, two questions:
1. **Forecasting** — rank currently-employed people by how likely they are to resign.
2. **Inference** — identify which factors actually drive that risk, and by how much.

An earlier version of this notebook ran a separate classifier (Logistic Regression / Random Forest / Gradient Boosting) to answer the forecasting side. Dropped: Cox PH already answers both questions on its own (`predict_partial_hazard`/`predict_survival_function` for forecasting, fitted hazard ratios for inference), so a second model with its own preprocessing/tuning/evaluation stack was pure overhead.

## Structure
```
code/
  hr_attrition_analysis.ipynb  <- the project: narrative, EDA, and the calls that drive it
  preprocessing.py             <- load_data, prepare_survival_data
  cox_model.py                 <- fit/validate/check-assumptions, hazard-ratio plots,
                                   retention curves, risk ranking, risk-distribution plot
  report_builder.py            <- executive PPTX design system + slide construction
```
The notebook stays readable as a sequence of `from <module> import ...` calls plus the numbers/decisions specific to this dataset (which covariates, which grouping factor for retention curves); the reusable logic behind each step lives in its module.

## Method
- Cox Proportional Hazards, validated with a train/test split (concordance index) before being refit on the full population for actual deployment scoring.
- Proportional-hazards assumption checked per covariate before trusting the hazard ratios.
- Hazard ratios shown two ways: the statistical-convention log-hazard forest plot, and a plain hazard-ratio bar chart on the actual HR scale — red where a factor raises resignation risk (HR > 1), green where it's protective (HR < 1). The second is what actually goes in the executive report, since "hazard ×2.2" reads far more directly than a log-coefficient.
- Kaplan-Meier retention curves + log-rank test on the strongest actionable factor (overtime).
- Forecasting output includes both a relative risk score and an absolute probability of leaving within 1 year (`predict_survival_function`).
- Results assembled into a 5-slide executive PowerPoint (Executive Summary + Model Validation, What Drives Attrition, Retention Curves, Who Is at Risk, Recommendations) by `report_builder.py`, reverse-engineered from a hand-refined version of the deck so every run reproduces the same look with fresh data and charts.

## Result
Concordance index ≈ 0.85 (test). Strongest driver: overtime (hazard ×2.2, p<0.001). 10 of 12 covariates statistically significant at p<0.05.

## How to run
Open `code/hr_attrition_analysis.ipynb` and run top to bottom. Data: `data/IBM-HR-Employee-Attrition.csv`. Outputs (figures + PPTX) are written to `result/`. Each run's report is saved as `Executive_Attrition_Report_<YYYYMMDD>.pptx` (a same-day re-run overwrites, by design); `Executive_Attrition_Report.pptx` (no date) is the original hand-refined reference `report_builder.py` was built from.
