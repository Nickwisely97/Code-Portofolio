"""
simulation_runner.py
Sweep and stress-test orchestration on top of queueing_model.simulate_pipeline --
runs multiple independent replications per arrival rate and pools them together
for a stable P99 estimate (a single run's tail is too noisy on its own).
"""

import pandas as pd

from queueing_model import (
    simulate_pipeline, simulate_combined_pipeline, simulate_operating_day, format_clock, seed_for,
)


def run_sweep(scenario_configs, kitchen_params, order_size_dist, lambdas, reps,
              sim_hours, warmup_hours, sla_seconds, gamma_shape, save_path=None):
    """For each (scenario, arrival rate) pair, pool `reps` replications and compute
    mean/P99/SLA% of total order-to-food time. Returns the sweep DataFrame."""
    rows = []
    for scenario, cfg in scenario_configs.items():
        for lam in lambdas:
            pooled = pd.concat([
                simulate_pipeline(lam, cfg, kitchen_params, order_size_dist, gamma_shape,
                                   sim_hours, warmup_hours, seed=seed_for(scenario, lam, r))
                for r in range(reps)
            ], ignore_index=True)
            rows.append({
                "scenario": scenario,
                "arrival_rate": lam,
                "n_customers": len(pooled),
                "mean_seconds": pooled["total"].mean(),
                "p99_seconds": pooled["total"].quantile(0.99),
                "sla_pct": (pooled["total"] <= sla_seconds).mean() * 100,
            })

    sweep_df = pd.DataFrame(rows)
    if save_path:
        sweep_df.to_csv(save_path, index=False)
    return sweep_df


def run_stress_test(scenario_configs, kitchen_params, order_size_dist, stress_lambda, reps,
                     sim_hours, warmup_hours, sla_seconds, gamma_shape, save_path=None):
    """Pool `reps` replications at a single stress-test arrival rate and break the
    total time down by stage. Returns the breakdown DataFrame."""
    rows = []
    for scenario, cfg in scenario_configs.items():
        pooled = pd.concat([
            simulate_pipeline(stress_lambda, cfg, kitchen_params, order_size_dist, gamma_shape,
                               sim_hours, warmup_hours, seed=seed_for(scenario, "stress", r))
            for r in range(reps)
        ], ignore_index=True)
        rows.append({
            "scenario": scenario,
            "order_wait": pooled["order_wait"].mean(),
            "order_service": pooled["order_service"].mean(),
            "kitchen_wait": pooled["kitchen_wait"].mean(),
            "kitchen_service": pooled["kitchen_service"].mean(),
            "p99_total": pooled["total"].quantile(0.99),
            "sla_pct": (pooled["total"] <= sla_seconds).mean() * 100,
        })

    breakdown_df = pd.DataFrame(rows)
    if save_path:
        breakdown_df.to_csv(save_path, index=False)
    return breakdown_df


def run_combined_sweep(cashier_cfg, kiosk_cfg, kitchen_params, order_size_dist, lambdas, reps,
                        sim_hours, warmup_hours, sla_seconds, gamma_shape, scenario_label,
                        save_path=None):
    """Same idea as run_sweep, but for the combined (both channels open, join-the-
    shorter-queue) configuration. Also tracks what share of customers ended up on
    each channel at each arrival rate."""
    rows = []
    for lam in lambdas:
        pooled = pd.concat([
            simulate_combined_pipeline(lam, cashier_cfg, kiosk_cfg, kitchen_params, order_size_dist,
                                        gamma_shape, sim_hours, warmup_hours,
                                        seed=seed_for(scenario_label, lam, r))
            for r in range(reps)
        ], ignore_index=True)
        rows.append({
            "scenario": scenario_label,
            "arrival_rate": lam,
            "n_customers": len(pooled),
            "mean_seconds": pooled["total"].mean(),
            "p99_seconds": pooled["total"].quantile(0.99),
            "sla_pct": (pooled["total"] <= sla_seconds).mean() * 100,
            "cashier_share_pct": (pooled["channel"] == "cashier").mean() * 100,
        })

    combined_df = pd.DataFrame(rows)
    if save_path:
        combined_df.to_csv(save_path, index=False)
    return combined_df


def run_kitchen_sizing(cashier_cfg, kiosk_cfg, kitchen_base_params, order_size_dist, target_lambda,
                        station_counts, reps, sim_hours, warmup_hours, sla_seconds, gamma_shape,
                        save_path=None):
    """Hold the combined front end fixed at target_lambda and sweep the number of
    kitchen stations, to find how many are needed before the kitchen stops being
    the bottleneck at that demand level."""
    rows = []
    for stations in station_counts:
        kitchen_cfg = {**kitchen_base_params, "servers": stations}
        pooled = pd.concat([
            simulate_combined_pipeline(target_lambda, cashier_cfg, kiosk_cfg, kitchen_cfg, order_size_dist,
                                        gamma_shape, sim_hours, warmup_hours,
                                        seed=seed_for("kitchen_sizing", stations, r))
            for r in range(reps)
        ], ignore_index=True)
        rows.append({
            "kitchen_stations": stations,
            "arrival_rate": target_lambda,
            "mean_seconds": pooled["total"].mean(),
            "p99_seconds": pooled["total"].quantile(0.99),
            "sla_pct": (pooled["total"] <= sla_seconds).mean() * 100,
        })

    sizing_df = pd.DataFrame(rows)
    if save_path:
        sizing_df.to_csv(save_path, index=False)
    return sizing_df


def run_operating_day(scenario_configs, kitchen_params, order_size_dist, gamma_shape,
                       rate_fn, open_hour, close_hour, save_path=None):
    """Simulate one full operating day per scenario and return a single combined,
    human-readable event log (one row per customer): which scenario, what time they
    arrived and were done (wall-clock), how many items, and time spent in each stage."""
    logs = []
    for scenario, cfg in scenario_configs.items():
        log = simulate_operating_day(cfg, kitchen_params, order_size_dist, gamma_shape,
                                      rate_fn, open_hour, close_hour, seed=seed_for(scenario, "operating_day"))
        log.insert(0, "scenario", scenario)
        logs.append(log)

    event_log = pd.concat(logs, ignore_index=True).sort_values(["scenario", "arrival_time"]).reset_index(drop=True)
    event_log.insert(0, "customer_id", event_log.groupby("scenario").cumcount() + 1)

    event_log["arrival_clock"] = event_log["arrival_time"].apply(lambda s: format_clock(s, open_hour))
    event_log["arrival_hour"] = (open_hour + event_log["arrival_time"] / 3600).round(2)
    event_log["completion_clock"] = event_log["completion_time"].apply(lambda s: format_clock(s, open_hour))
    event_log["queue_wait_seconds"] = event_log["order_wait"] + event_log["kitchen_wait"]

    event_log = event_log.rename(columns={
        "order_wait": "order_queue_wait_seconds",
        "order_service": "ordering_time_seconds",
        "kitchen_wait": "kitchen_queue_wait_seconds",
        "kitchen_service": "food_prep_time_seconds",
        "total_time": "total_time_seconds",
    })
    event_log = event_log[[
        "customer_id", "scenario", "arrival_clock", "arrival_hour", "items",
        "order_queue_wait_seconds", "ordering_time_seconds",
        "kitchen_queue_wait_seconds", "food_prep_time_seconds",
        "queue_wait_seconds", "total_time_seconds", "completion_clock",
    ]]

    if save_path:
        event_log.to_csv(save_path, index=False)
    return event_log
