"""
plotting.py
Chart functions for the SOT vs. cashier staffing analysis, in the McDonald's
brand palette (black/navy, red, yellow) used across this project's figures.
"""

import numpy as np
import matplotlib.pyplot as plt

MCD_RED = "#DA291C"
MCD_YELLOW = "#FFC72C"
MCD_BLACK = "#27251F"
COMBINED_BLUE = "#4C72B0"
DEFAULT_COLORS = [MCD_BLACK, MCD_RED, COMBINED_BLUE]


def plot_order_size_distribution(order_size_dist, save_path):
    mean_items = (order_size_dist["items"] * order_size_dist["probability"]).sum()
    plt.figure(figsize=(6, 4))
    plt.bar(order_size_dist["items"], order_size_dist["probability"] * 100, color=MCD_YELLOW, edgecolor=MCD_BLACK)
    plt.xlabel("Items per Order")
    plt.ylabel("Probability (%)")
    plt.title(f"Order-Size Distribution (mean = {mean_items:.2f} items)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_capacity_ceiling(ceiling_df, save_path):
    plt.figure(figsize=(8, 5))
    colors = [MCD_BLACK, MCD_RED, MCD_YELLOW]
    plt.bar(ceiling_df["resource"], ceiling_df["max_stable_throughput"], color=colors, edgecolor=MCD_BLACK)
    for i, v in enumerate(ceiling_df["max_stable_throughput"]):
        plt.text(i, v + 1, f"{v:.0f}/hr", ha="center", fontweight="bold")
    plt.ylabel("Max Stable Throughput (customers/hour)")
    plt.title("Capacity Ceiling by Resource")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_p99_vs_arrival_rate(sweep_df, scenarios, sla_seconds, save_path, ylim=2500, colors=None, title=None):
    colors = colors or DEFAULT_COLORS[:len(scenarios)]
    plt.figure(figsize=(9, 6))
    for scenario, color in zip(scenarios, colors):
        subset = sweep_df[sweep_df["scenario"] == scenario]
        plt.plot(subset["arrival_rate"], subset["p99_seconds"], label=scenario, color=color, linewidth=2)

    plt.axhline(sla_seconds, color=MCD_YELLOW, linestyle="--",
                label=f"SLA target ({sla_seconds:.0f}s / {sla_seconds / 60:.0f} min)")
    plt.ylim(0, ylim)
    plt.xlabel("Arrival Rate (customers/hour)")
    plt.ylabel("P99 Total Time: Order to Food (seconds)")
    plt.title(title or "P99 End-to-End Time vs. Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_sla_vs_arrival_rate(sweep_df, scenarios, save_path, colors=None, title=None):
    colors = colors or DEFAULT_COLORS[:len(scenarios)]
    plt.figure(figsize=(9, 6))
    for scenario, color in zip(scenarios, colors):
        subset = sweep_df[sweep_df["scenario"] == scenario]
        plt.plot(subset["arrival_rate"], subset["sla_pct"], label=scenario, color=color, linewidth=2)

    plt.axhline(99, color=MCD_YELLOW, linestyle="--", label="99% target (= P99 SLA)")
    plt.xlabel("Arrival Rate (customers/hour)")
    plt.ylabel("% Served Within 10 Minutes")
    plt.title(title or "Service Level vs. Demand: Both Scenarios Capped by the Same Kitchen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_kitchen_sizing(sizing_df, sla_seconds, target_lambda, save_path):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(sizing_df["kitchen_stations"], sizing_df["p99_seconds"], color=MCD_RED,
             linewidth=2, marker="o", label="P99 total time")
    ax1.axhline(sla_seconds, color=MCD_YELLOW, linestyle="--", label=f"SLA target ({sla_seconds / 60:.0f} min)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Kitchen Stations")
    ax1.set_ylabel("P99 Total Time: Order to Food (seconds, log scale)")
    ax1.set_title(f"Kitchen Sizing for the Combined Front End at {target_lambda} Customers/Hour")
    ax1.set_xticks(sizing_df["kitchen_stations"])
    ax1.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_workforce_efficiency(workforce_df, save_path, colors=None):
    plt.figure(figsize=(9, 5.5))
    colors = colors or (DEFAULT_COLORS * 2)[:len(workforce_df)]
    bars = plt.bar(workforce_df["configuration"], workforce_df["throughput_per_staff"], color=colors, edgecolor=MCD_BLACK)
    for bar, staff, cap in zip(bars, workforce_df["total_staff"], workforce_df["p99_safe_capacity"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                  f"{staff:.0f} staff\n{cap:.0f}/hr", ha="center", fontsize=10, fontweight="bold")
    plt.ylim(0, workforce_df["throughput_per_staff"].max() * 1.3)
    plt.ylabel("P99-Safe Capacity per Staff Member (customers/hour/person)")
    plt.title("Workforce Efficiency by Configuration")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_demand_profile(rate_fn, open_hour, close_hour, save_path):
    hours = np.linspace(open_hour, close_hour, 200)
    rates = [rate_fn(h) for h in hours]

    plt.figure(figsize=(9, 5))
    plt.plot(hours, rates, color=MCD_RED, linewidth=2.5)
    plt.fill_between(hours, rates, color=MCD_YELLOW, alpha=0.3)
    plt.xlabel("Time of Day")
    plt.ylabel("Arrival Rate (customers/hour)")
    plt.title("Daily Demand Profile: Lunch and After-Office Peaks")
    tick_hours = list(range(int(open_hour), int(close_hour) + 1, 2))
    plt.xticks(tick_hours, [f"{h % 24:02d}:00" for h in tick_hours])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_hourly_wait_time(event_log, scenarios, sla_seconds, save_path):
    plt.figure(figsize=(9, 6))
    for scenario, color in zip(scenarios, [MCD_BLACK, MCD_RED]):
        subset = event_log[event_log["scenario"] == scenario]
        hourly = subset.groupby(subset["arrival_hour"].astype(int))["total_time_seconds"].mean()
        plt.plot(hourly.index, hourly.values, label=scenario, color=color, linewidth=2, marker="o", markersize=4)

    plt.axhline(sla_seconds, color=MCD_YELLOW, linestyle="--", label=f"SLA target ({sla_seconds / 60:.0f} min)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Total Time: Order to Food (seconds)")
    plt.title("Mean Wait Time by Hour -- One Simulated Operating Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path


def plot_stress_test_breakdown(breakdown_df, sla_seconds, stress_lambda, save_path):
    stages = ["order_wait", "order_service", "kitchen_wait", "kitchen_service"]
    stage_labels = ["Order Queue", "Order Service", "Kitchen Queue", "Kitchen Prep"]
    stage_colors = [MCD_YELLOW, MCD_BLACK, "#8C8C8C", MCD_RED]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(breakdown_df))
    for stage, label, color in zip(stages, stage_labels, stage_colors):
        ax.bar(breakdown_df["scenario"], breakdown_df[stage], bottom=bottom, label=label, color=color, edgecolor=MCD_BLACK)
        bottom += breakdown_df[stage].values

    ax.axhline(sla_seconds, color="grey", linestyle="--", label=f"{sla_seconds / 60:.0f}-min SLA")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title(f"Where the Time Goes at {stress_lambda} Customers/Hour")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    return save_path
