"""
cox_model.py
Cox Proportional Hazards fitting, validation, inference, and forecasting
for the Headcount Attrition analysis.
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

RISK_COLOR = "#B0413E"
PROTECTIVE_COLOR = "#2E7D5B"


def validate(surv_df, penalizer=0.1, test_size=0.2, random_state=42):
    """Fit on a train split, score on the held-out test split. Returns
    (cph_val, surv_train, surv_test, test_cindex)."""
    surv_train, surv_test = train_test_split(
        surv_df, test_size=test_size, random_state=random_state, stratify=surv_df["event"]
    )
    cph_val = CoxPHFitter(penalizer=penalizer)
    cph_val.fit(surv_train, duration_col="duration", event_col="event")
    test_cindex = concordance_index(
        surv_test["duration"], -cph_val.predict_partial_hazard(surv_test), surv_test["event"]
    )
    return cph_val, surv_train, surv_test, test_cindex


def check_ph_assumption(cph_val, surv_train, p_value_threshold=0.05):
    """Print lifelines' proportional-hazards diagnostic for the validation model."""
    return cph_val.check_assumptions(surv_train, p_value_threshold=p_value_threshold, show_plots=False)


def fit_full(surv_df, penalizer=0.1):
    """Refit on the full population -- the model actually used for deployment scoring."""
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(surv_df, duration_col="duration", event_col="event")
    return cph


def get_significant_factors(cph, alpha=0.05):
    """Returns (cox_summary, significant) -- full hazard-ratio table and the
    p < alpha subset, sorted by p-value."""
    cox_summary = cph.summary.sort_values("exp(coef)", ascending=False)
    significant = cox_summary[cox_summary["p"] < alpha].sort_values("p")
    return cox_summary, significant


def plot_hazard_ratio_forest(cph, save_path):
    """Lifelines' log-hazard-ratio forest plot (coefficient scale, with 95% CI)."""
    plt.figure(figsize=(8, 6))
    cph.plot()
    plt.title("Cox Model -- Log Hazard Ratios (95% CI)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return save_path


def plot_hazard_ratios_bar(cox_summary, save_path):
    """Hazard-ratio bar chart on the actual HR scale (not log) -- the number an
    executive reads directly. Red bars raise resignation risk (HR > 1), green
    bars are protective (HR < 1); the dashed line at 1.0 marks no effect."""
    plot_df = cox_summary.sort_values("exp(coef)")
    colors = [RISK_COLOR if hr > 1 else PROTECTIVE_COLOR for hr in plot_df["exp(coef)"]]

    plt.figure(figsize=(9, 6))
    bars = plt.barh(plot_df.index, plot_df["exp(coef)"], color=colors)
    plt.axvline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Hazard Ratio (1.0 = no effect on resignation risk)")
    plt.title("Hazard Ratio by Factor -- Red Raises Risk, Green Lowers It", fontsize=13)
    for bar, hr in zip(bars, plot_df["exp(coef)"]):
        plt.text(hr, bar.get_y() + bar.get_height() / 2, f" {hr:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return save_path


def compute_retention_curves(df, group_col, duration_col, target_col, save_path):
    """Kaplan-Meier retention curves split by group_col (Yes/No), plus a log-rank
    test between the two groups. Saves the plot and returns the log-rank result."""
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, mask in [(f"No {group_col}", df[group_col] == "No"), (group_col, df[group_col] == "Yes")]:
        kmf.fit(
            durations=df.loc[mask, duration_col].clip(lower=0.5),
            event_observed=(df.loc[mask, target_col] == "Yes").astype(int),
            label=label,
        )
        kmf.plot_survival_function(ax=ax)

    plt.title(f"Retention Curve by {group_col} Status", fontsize=13)
    plt.xlabel(duration_col)
    plt.ylabel("Probability of Still Being Employed")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    mask = df[group_col] == "Yes"
    lr = logrank_test(
        df.loc[mask, duration_col], df.loc[~mask, duration_col],
        event_observed_A=(df.loc[mask, target_col] == "Yes").astype(int),
        event_observed_B=(df.loc[~mask, target_col] == "Yes").astype(int),
    )
    print(f"Log-rank test ({group_col} vs No {group_col}): p = {lr.p_value:.2e}")
    return lr


def rank_current_employees(cph, surv_df, employee_ids, top_n=15):
    """Score every still-employed row: relative risk, predicted median tenure,
    and probability of leaving within 1 year. Returns (current, at_risk)."""
    current = surv_df[surv_df["event"] == 0].copy()
    current["Employee ID"] = employee_ids.loc[current.index].values
    current["Risk Score"] = cph.predict_partial_hazard(current)
    current["Predicted Median Tenure (Years)"] = cph.predict_median(current)

    survival_at_1y = cph.predict_survival_function(current, times=[1.0])
    current["P(Leave within 1 Year)"] = 1 - survival_at_1y.loc[1.0]

    at_risk = current.sort_values("Risk Score", ascending=False).head(top_n)
    return current, at_risk


def plot_risk_distribution(current, at_risk, save_path):
    """Risk-score histogram across all current employees, plus a bar chart of the
    top at-risk employees by 1-year leave probability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(current["Risk Score"], bins=30, color="#4C72B0", alpha=0.85)
    axes[0].axvline(current["Risk Score"].median(), color="black", linestyle="--", label="Median")
    axes[0].set_title("Risk Score Distribution -- Current Employees")
    axes[0].set_xlabel("Risk Score (relative hazard)")
    axes[0].set_ylabel("Employees")
    axes[0].legend()

    top_sorted = at_risk.sort_values("P(Leave within 1 Year)")
    axes[1].barh(top_sorted["Employee ID"].astype(str), top_sorted["P(Leave within 1 Year)"], color="#C44E52")
    axes[1].set_title(f"Top {len(at_risk)} At-Risk Employees")
    axes[1].set_xlabel("P(Leave within 1 Year)")
    axes[1].set_ylabel("Employee ID")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return save_path
