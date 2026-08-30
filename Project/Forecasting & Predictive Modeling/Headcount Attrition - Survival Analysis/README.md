# Headcount Attrition - Survival Analysis

**CV skill represented:** Survival & Reliability Analysis (Cox Proportional Hazards), Applied Statistics (log-rank test, proportional-hazards assumption check).

## Problem
One model, two questions:
1. **Forecasting** — rank currently-employed people by how likely they are to resign.
2. **Inference** — identify which factors actually drive that risk, and by how much.

An earlier version of this notebook ran a separate classifier (Logistic Regression / Random Forest / Gradient Boosting) to answer the forecasting side. Dropped: Cox PH already answers both questions on its own (`predict_partial_hazard`/`predict_survival_function` for forecasting, fitted hazard ratios for inference), so a second model with its own preprocessing/tuning/evaluation stack was pure overhead.

## Method
- Cox Proportional Hazards, validated with a train/test split (concordance index) before being refit on the full population for actual deployment scoring.
- Proportional-hazards assumption checked per covariate before trusting the hazard ratios.
- Kaplan-Meier retention curves + log-rank test on the strongest actionable factor (overtime).
- Forecasting output includes both a relative risk score and an absolute probability of leaving within 1 year (`predict_survival_function`).
- Results assembled into a 5-slide executive PowerPoint (Executive Summary + Model Validation, What Drives Attrition, Retention Curves, Who Is at Risk, Recommendations) built from a reusable design system (STEP 11) — colors, type scale, card/panel components, header/footer — reverse-engineered from a hand-refined version of the deck, so every run reproduces the same look with fresh data and charts.

## Result
Concordance index ≈ 0.85 (test). Strongest driver: overtime (hazard ×2.2, p<0.001). 10 of 12 covariates statistically significant at p<0.05.

## How to run
Open `code/hr_attrition_analysis.ipynb` and run top to bottom. Data: `data/IBM-HR-Employee-Attrition.csv`. Outputs (figures + PPTX) are written to `result/`. Each run's report is saved as `Executive_Attrition_Report_<timestamp>.pptx` so re-running never overwrites a previous version; `Executive_Attrition_Report.pptx` (no timestamp) is the original hand-refined reference the design system in STEP 11 was built from.
