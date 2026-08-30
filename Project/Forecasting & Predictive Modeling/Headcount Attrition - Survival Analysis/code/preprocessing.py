"""
preprocessing.py
Data loading and survival-format preparation for the Headcount Attrition analysis.
"""

import re
import pandas as pd


def clean_feature_label(name):
    name = name.replace('_', ' ')
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return name.replace(' Yes', ' (Yes)').replace(' No', ' (No)').strip().title()


def load_data(data_path):
    """Load the raw employee CSV. Returns (df, employee_ids)."""
    df = pd.read_csv(data_path)
    employee_ids = df["EmployeeNumber"].copy()
    return df, employee_ids


def prepare_survival_data(df, covariates, dummy_cols, duration_col="YearsAtCompany", target_col="Attrition"):
    """Build a Cox-PH-ready frame: duration/event columns plus the given covariates,
    with dummy_cols one-hot encoded and every column renamed to a readable label."""
    surv_df = df[[duration_col, target_col] + covariates].copy()
    surv_df["event"] = (surv_df[target_col] == "Yes").astype(int)
    surv_df["duration"] = surv_df[duration_col].clip(lower=0.5)
    surv_df = surv_df.drop(columns=[target_col, duration_col])

    surv_df = pd.get_dummies(surv_df, columns=dummy_cols, drop_first=True)
    surv_df.columns = [c if c in ("duration", "event") else clean_feature_label(c) for c in surv_df.columns]
    return surv_df.astype(float)
