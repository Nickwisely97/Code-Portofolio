"""
queueing_model.py
Two-stage (order-taking -> kitchen) discrete-event queueing simulation and
capacity math for the SOT vs. cashier staffing analysis.
"""

import zlib

import numpy as np
import pandas as pd
import simpy


def seed_for(*parts):
    """Deterministic seed from arbitrary parts -- reruns reproduce identical results
    (unlike Python's built-in hash(), which is randomized per process for strings)."""
    return zlib.crc32("_".join(map(str, parts)).encode())


def draw_items(rng, order_size_dist, n=1):
    return rng.choice(order_size_dist["items"].values, size=n, p=order_size_dist["probability"].values)


def gamma_time(rng, mean, shape):
    return rng.gamma(shape, mean / shape)


def simulate_pipeline(lam, order_cfg, kitchen_params, order_size_dist, gamma_shape,
                       sim_hours, warmup_hours, seed):
    """Simulate the order -> kitchen tandem queue for one scenario at arrival rate
    lam (customers/hour). Order size is drawn once per customer and drives the
    service time at both stages. Returns a DataFrame of per-customer stage times
    (post-warmup only): order_wait, order_service, kitchen_wait, kitchen_service, total."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    order_stage = simpy.Resource(env, capacity=int(order_cfg["servers"]))
    kitchen = simpy.Resource(env, capacity=int(kitchen_params["servers"]))
    records = []

    def customer(arrival_time):
        items = draw_items(rng, order_size_dist, 1)[0]
        order_mean = order_cfg["aht_base_seconds"] + order_cfg["aht_per_item_seconds"] * items
        kitchen_mean = kitchen_params["base_seconds"] + kitchen_params["per_item_seconds"] * items

        with order_stage.request() as req:
            yield req
            order_wait = env.now - arrival_time
            order_service = gamma_time(rng, order_mean, gamma_shape)
            yield env.timeout(order_service)

        kitchen_arrive = env.now
        with kitchen.request() as req2:
            yield req2
            kitchen_wait = env.now - kitchen_arrive
            kitchen_service = gamma_time(rng, kitchen_mean, gamma_shape)
            yield env.timeout(kitchen_service)

        if arrival_time / 3600 >= warmup_hours:
            records.append((order_wait, order_service, kitchen_wait, kitchen_service,
                             env.now - arrival_time))

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(3600 / lam))
            env.process(customer(env.now))

    env.process(arrivals())
    env.run(until=sim_hours * 3600)
    return pd.DataFrame(records, columns=["order_wait", "order_service", "kitchen_wait",
                                           "kitchen_service", "total"])


def simulate_combined_pipeline(lam, cashier_cfg, kiosk_cfg, kitchen_params, order_size_dist, gamma_shape,
                                sim_hours, warmup_hours, seed):
    """Both order-taking channels running at once, sharing one arrival stream and
    one downstream kitchen. Each customer joins whichever channel has fewer people
    in it right now (join-the-shorter-queue -- what a customer glancing at both
    lines would actually do), then both channels feed the same shared kitchen.
    Returns the same per-customer columns as simulate_pipeline, plus 'channel'."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    cashier_stage = simpy.Resource(env, capacity=int(cashier_cfg["servers"]))
    kiosk_stage = simpy.Resource(env, capacity=int(kiosk_cfg["servers"]))
    kitchen = simpy.Resource(env, capacity=int(kitchen_params["servers"]))
    records = []

    def in_system(resource):
        return resource.count + len(resource.queue)

    def customer(arrival_time):
        items = draw_items(rng, order_size_dist, 1)[0]

        if in_system(cashier_stage) <= in_system(kiosk_stage):
            channel, stage, cfg = "cashier", cashier_stage, cashier_cfg
        else:
            channel, stage, cfg = "kiosk", kiosk_stage, kiosk_cfg

        order_mean = cfg["aht_base_seconds"] + cfg["aht_per_item_seconds"] * items
        kitchen_mean = kitchen_params["base_seconds"] + kitchen_params["per_item_seconds"] * items

        with stage.request() as req:
            yield req
            order_wait = env.now - arrival_time
            order_service = gamma_time(rng, order_mean, gamma_shape)
            yield env.timeout(order_service)

        kitchen_arrive = env.now
        with kitchen.request() as req2:
            yield req2
            kitchen_wait = env.now - kitchen_arrive
            kitchen_service = gamma_time(rng, kitchen_mean, gamma_shape)
            yield env.timeout(kitchen_service)

        if arrival_time / 3600 >= warmup_hours:
            records.append((channel, order_wait, order_service, kitchen_wait, kitchen_service,
                             env.now - arrival_time))

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(3600 / lam))
            env.process(customer(env.now))

    env.process(arrivals())
    env.run(until=sim_hours * 3600)
    return pd.DataFrame(records, columns=["channel", "order_wait", "order_service", "kitchen_wait",
                                           "kitchen_service", "total"])


def format_clock(seconds_since_open, open_hour):
    """Convert seconds since opening into a wall-clock 'HH:MM:SS' string."""
    total_seconds = int(round(open_hour * 3600 + seconds_since_open))
    hh = (total_seconds // 3600) % 24
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def arrival_rate_profile(clock_hour, baseline, lunch_peak, lunch_center, lunch_width,
                          dinner_peak, dinner_center, dinner_width):
    """Customers/hour at a given clock hour: a low baseline (early morning, late
    evening) plus two Gaussian-shaped bumps -- a taller lunch peak and a shorter
    after-office/dinner peak."""
    lunch = lunch_peak * np.exp(-0.5 * ((clock_hour - lunch_center) / lunch_width) ** 2)
    dinner = dinner_peak * np.exp(-0.5 * ((clock_hour - dinner_center) / dinner_width) ** 2)
    return baseline + lunch + dinner


def simulate_operating_day(order_cfg, kitchen_params, order_size_dist, gamma_shape,
                            rate_fn, open_hour, close_hour, seed, drain_buffer_hours=2):
    """Simulate one full operating day (open_hour to close_hour) with a time-varying
    arrival rate given by rate_fn(clock_hour), using Poisson thinning: draw candidate
    arrivals at the day's peak rate, then keep each with probability
    rate_fn(t) / peak_rate. New arrivals stop at closing; the sim keeps running for
    drain_buffer_hours so customers already queued at close still get served.

    Returns every customer's full event record (no warm-up discard -- this is a
    literal one-day log, not a steady-state estimate): arrival_time, items,
    order_wait, order_service, kitchen_wait, kitchen_service, completion_time."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    order_stage = simpy.Resource(env, capacity=int(order_cfg["servers"]))
    kitchen = simpy.Resource(env, capacity=int(kitchen_params["servers"]))
    records = []

    operating_seconds = (close_hour - open_hour) * 3600
    grid = np.linspace(open_hour, close_hour, 200)
    peak_rate = max(rate_fn(h) for h in grid)

    def customer(arrival_time, items):
        order_mean = order_cfg["aht_base_seconds"] + order_cfg["aht_per_item_seconds"] * items
        kitchen_mean = kitchen_params["base_seconds"] + kitchen_params["per_item_seconds"] * items

        with order_stage.request() as req:
            yield req
            order_wait = env.now - arrival_time
            order_service = gamma_time(rng, order_mean, gamma_shape)
            yield env.timeout(order_service)

        kitchen_arrive = env.now
        with kitchen.request() as req2:
            yield req2
            kitchen_wait = env.now - kitchen_arrive
            kitchen_service = gamma_time(rng, kitchen_mean, gamma_shape)
            yield env.timeout(kitchen_service)

        records.append((arrival_time, items, order_wait, order_service, kitchen_wait,
                         kitchen_service, env.now))

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(3600 / peak_rate))
            if env.now > operating_seconds:
                break
            clock_hour = open_hour + env.now / 3600
            if rng.random() <= rate_fn(clock_hour) / peak_rate:
                items = draw_items(rng, order_size_dist, 1)[0]
                env.process(customer(env.now, items))

    env.process(arrivals())
    env.run(until=operating_seconds + drain_buffer_hours * 3600)

    log = pd.DataFrame(records, columns=["arrival_time", "items", "order_wait", "order_service",
                                          "kitchen_wait", "kitchen_service", "completion_time"])
    log["total_time"] = log["completion_time"] - log["arrival_time"]
    return log.sort_values("arrival_time").reset_index(drop=True)


def theoretical_ceiling(servers, base_seconds, per_item_seconds, mean_items):
    """Max stable throughput (customers/hour) for one resource on its own:
    servers x 3600 / mean_service_seconds."""
    mean_service = base_seconds + per_item_seconds * mean_items
    return servers * 3600 / mean_service


def find_crossing(df, threshold, value_col="p99_seconds", x_col="arrival_rate"):
    """Linear interpolation for the arrival rate where value_col first crosses
    threshold, between the two bracketing grid points."""
    df = df.sort_values(x_col).reset_index(drop=True)
    for i in range(len(df) - 1):
        y0, y1 = df.loc[i, value_col], df.loc[i + 1, value_col]
        if y0 <= threshold < y1:
            x0, x1 = df.loc[i, x_col], df.loc[i + 1, x_col]
            return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return np.nan
