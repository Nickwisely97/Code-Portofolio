"""
preprocessing.py
Feature engineering for Indonesian temperature forecasting.

Pipeline built around the forecast-origin concept T0: the model may only use
information known at/before T0, then predicts every hour from T0+1h through
T0+horizon_hours (default 336h / 14 days) at once (direct multi-horizon),
not autoregressively hour by hour.

Two feature groups:
  1. Origin state — computed once per origin from data <= T0:
       temperature lags (t-1, t-3, t-6, t-12, t-24, t-48, t-72h)
       rolling temperature stats (mean/std/min/max, 7/14/30/60/90-day windows)
       temperature momentum (t-6h, t-24h differences)
       humidity/wind_speed/precipitation at origin + lag-1h/lag-24h
     These values are broadcast across every horizon row of that origin.
  2. Target calendar — computed from the TARGET timestamp (T0+h), not T0,
     since calendar facts (hour, day, month, season) for any future date
     are always knowable in advance, so they're safe to use.

Plus `horizon_h` (hours ahead of the origin) and climatology (historical
mean/std temperature per (month, day), computed only from data <= a given
cutoff) as extra signal for far-out horizons.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────
LAG_HOURS              = [1, 3, 6, 12, 24, 48, 72]
ROLLING_DAYS           = [7, 14, 30, 60, 90]          # in days -> ×24 hours
MOMENTUM_HOURS         = [6, 24]
WEATHER_AUX_COLS       = ["humidity", "wind_speed", "precipitation"]
TARGET_COL             = "temperature"
ORIGIN_STRIDE_HOURS    = 24     # spacing between forecast origins when sampling
FORECAST_HORIZON_HOURS = 336    # 14 days


# ─── Helper: Cyclical Encoding ───────────────────────────────────────────────
def _cyclic(series: pd.Series, max_val: float):
    """Encode a series as a (sin, cos) pair for a cyclical representation."""
    angle = 2 * np.pi * series / max_val
    return np.sin(angle), np.cos(angle)


# ─── Target-side calendar features ───────────────────────────────────────────
def build_temporal_features(ts: pd.Series) -> pd.DataFrame:
    """
    Cyclical calendar features for a timestamp Series (applied to the
    TARGET timestamp when building the direct-horizon frame -- not the
    origin -- since calendar facts for any date are always knowable
    in advance).

    Args:
        ts : datetime Series.

    Returns:
        DataFrame of calendar columns, same index as `ts`.
    """
    ts = pd.to_datetime(ts)
    out = pd.DataFrame(index=ts.index)

    sin_h, cos_h = _cyclic(ts.dt.hour, 24)
    out["sin_hour"], out["cos_hour"] = sin_h, cos_h

    sin_dow, cos_dow = _cyclic(ts.dt.dayofweek, 7)
    out["sin_dayofweek"], out["cos_dayofweek"] = sin_dow, cos_dow

    sin_dom, cos_dom = _cyclic(ts.dt.day, 31)
    out["sin_dayofmonth"], out["cos_dayofmonth"] = sin_dom, cos_dom

    sin_m, cos_m = _cyclic(ts.dt.month, 12)
    out["sin_month"], out["cos_month"] = sin_m, cos_m

    sin_doy, cos_doy = _cyclic(ts.dt.dayofyear, 366)
    out["sin_dayofyear"], out["cos_dayofyear"] = sin_doy, cos_doy

    out["hour"]            = ts.dt.hour.astype(np.int8)
    out["dayofweek"]       = ts.dt.dayofweek.astype(np.int8)
    out["month"]           = ts.dt.month.astype(np.int8)
    out["year"]            = ts.dt.year.astype(np.int16)
    out["quarter"]         = ts.dt.quarter.astype(np.int8)
    out["is_weekend"]      = (ts.dt.dayofweek >= 5).astype(np.int8)
    out["is_rainy_season"] = ts.dt.month.isin([11, 12, 1, 2, 3]).astype(np.int8)

    return out


# ─── Origin-state features ───────────────────────────────────────────────────
def build_lag_features(df: pd.DataFrame, lags: List[int] = LAG_HOURS) -> pd.DataFrame:
    """
    Temperature lags relative to each row's own timestamp. Used as origin
    state: a row at T0 represents the temperature already known N hours
    before T0.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[TARGET_COL].shift(lag)
    logger.debug(f"Lag features ({lags}) done.")
    return df


def build_rolling_features(df: pd.DataFrame, windows_days: List[int] = ROLLING_DAYS) -> pd.DataFrame:
    """
    Rolling mean/std/min/max temperature, computed from data BEFORE each
    row's timestamp (shift(1) first) -- origin state, not future info.
    min_periods = 50% of the window size so the start of the series isn't
    mostly NaN.
    """
    df = df.copy()
    for days in windows_days:
        w     = days * 24
        min_p = max(1, w // 2)
        rolled = df[TARGET_COL].shift(1).rolling(window=w, min_periods=min_p)
        df[f"roll_mean_{days}d"] = rolled.mean()
        df[f"roll_std_{days}d"]  = rolled.std()
        df[f"roll_min_{days}d"]  = rolled.min()
        df[f"roll_max_{days}d"]  = rolled.max()
    logger.debug(f"Rolling features ({windows_days} days) done.")
    return df


def build_momentum_features(df: pd.DataFrame, periods: List[int] = MOMENTUM_HOURS) -> pd.DataFrame:
    """Momentum = temperature difference vs. t-n hours ago (direction & speed of change)."""
    df = df.copy()
    for p in periods:
        df[f"momentum_{p}h"] = df[TARGET_COL].diff(p)
    logger.debug(f"Momentum features ({periods}) done.")
    return df


def build_weather_features(df: pd.DataFrame, aux_cols: List[str] = WEATHER_AUX_COLS) -> pd.DataFrame:
    """
    Extra weather features as origin state: humidity/wind_speed/precipitation
    AT the origin (`{col}_at_origin`) plus lag-1h and lag-24h. The raw
    (contemporaneous-with-the-row) columns are DROPPED from the final
    result -- for horizons beyond a few hours, "current" weather isn't
    knowable, only historical values up to the origin are fair game.
    """
    df = df.copy()
    present = [c for c in aux_cols if c in df.columns]

    for col in present:
        filled = df[col].ffill().bfill()
        df[f"{col}_at_origin"] = filled
        df[f"{col}_lag1"]      = filled.shift(1)
        df[f"{col}_lag24"]     = filled.shift(24)

    df = df.drop(columns=present)
    logger.debug("Weather auxiliary (origin-state) features done.")
    return df


def build_origin_state(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all the origin-state builders above. The resulting row, at
    timestamp T0, represents "what was already known at T0" -- reused
    (broadcast) across every horizon of that origin T0, not treated as a
    feature belonging to row T0 itself.
    """
    df = build_lag_features(df_hourly)
    df = build_rolling_features(df)
    df = build_momentum_features(df)
    df = build_weather_features(df)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a regular hourly (1H) frequency. Gaps are
    filled via linear interpolation for temperature, forward-fill for
    everything else.
    """
    df = df.set_index("timestamp").sort_index()
    df_hourly = df.resample("1h").mean(numeric_only=True)

    if "station" in df.columns:
        df_hourly["station"] = df["station"].iloc[0]

    if TARGET_COL in df_hourly.columns:
        df_hourly[TARGET_COL] = df_hourly[TARGET_COL].interpolate(method="time", limit=6)

    df_hourly = df_hourly.ffill(limit=6)

    df_hourly = df_hourly.reset_index().rename(columns={"index": "timestamp"})
    logger.info(f"Resampled to hourly: {len(df_hourly)} rows.")
    return df_hourly


def run_preprocessing(
    df: pd.DataFrame,
    station_name: Optional[str] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build the full origin-state table from raw data.

    Args:
        df           : Raw DataFrame with 'timestamp' and 'temperature' columns.
        station_name : Station name (optional, for logging & the 'station' column).
        drop_na      : Drop rows whose state is still NaN (too early in
                        history for the largest lag/rolling window) -- these
                        rows can't be used as an origin.

    Returns:
        Origin-state DataFrame: timestamp, temperature, [station], plus
        lag/rolling/momentum/weather features. (Calendar features are NOT
        here -- those are computed later for the target timestamp, see
        build_direct_horizon_frame.)
    """
    label = station_name or "unknown"
    logger.info(f"[Preprocessing] Starting for station: {label} | initial rows: {len(df)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = resample_to_hourly(df)
    df = build_origin_state(df)

    if station_name and "station" not in df.columns:
        df["station"] = station_name

    rows_before = len(df)
    if drop_na:
        state_cols = get_feature_columns(df)
        df = df.dropna(subset=state_cols).reset_index(drop=True)
    rows_after = len(df)
    logger.info(f"[Preprocessing] Done: {rows_after}/{rows_before} valid origin rows.")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature columns, excluding bookkeeping/target
    columns: timestamp, station, source, origin, temperature, weather_code.
    Used for both the origin-state table and the direct-horizon frame.
    """
    exclude = {"timestamp", "station", "source", "origin", TARGET_COL, "weather_code"}
    return [c for c in df.columns if c not in exclude]


# ─── Climatology ──────────────────────────────────────────────────────────────
def build_climatology(df_actual: pd.DataFrame, as_of) -> pd.DataFrame:
    """
    Historical mean & std temperature per (month, day), computed ONLY from
    data on/before `as_of` -- keeps this leakage-safe when used as a
    feature for any origin >= as_of.

    Returns:
        DataFrame with columns: month, day, climatology_mean, climatology_std.
    """
    as_of = pd.to_datetime(as_of)
    hist = df_actual[df_actual["timestamp"] <= as_of].copy()
    hist["month"] = hist["timestamp"].dt.month
    hist["day"]   = hist["timestamp"].dt.day

    clim = (hist.groupby(["month", "day"])[TARGET_COL]
                .agg(climatology_mean="mean", climatology_std="std")
                .reset_index())

    # Fallback for (month, day) buckets with too few samples (std NaN).
    clim["climatology_mean"] = clim["climatology_mean"].fillna(clim["climatology_mean"].mean())
    clim["climatology_std"]  = clim["climatology_std"].fillna(clim["climatology_std"].mean())

    logger.info(f"[Climatology] Built from data <= {as_of.date()}: {len(clim)} (month, day) buckets.")
    return clim


# ─── Origin selection ──────────────────────────────────────────────────────────
def select_origins(
    df_state: pd.DataFrame,
    start=None,
    end=None,
    stride_hours: int = ORIGIN_STRIDE_HOURS,
) -> pd.Series:
    """
    Pick timestamps fit to use as a forecast origin: state is complete (no
    NaN), falls within [start, end], and is spaced `stride_hours` apart
    (default: daily) so the number of origins stays manageable when
    building the training/backtest set.
    """
    state_cols = get_feature_columns(df_state)
    df = df_state.dropna(subset=state_cols)

    ts = df["timestamp"]
    if start is not None:
        ts = ts[ts >= pd.to_datetime(start)]
    if end is not None:
        ts = ts[ts <= pd.to_datetime(end)]

    ts = ts.sort_values().reset_index(drop=True)
    if stride_hours > 1:
        ts = ts.iloc[::stride_hours].reset_index(drop=True)

    return ts


# ─── Direct multi-horizon frame ──────────────────────────────────────────────
def build_direct_horizon_frame(
    df_state: pd.DataFrame,
    df_actual: pd.DataFrame,
    origins,
    climatology_table: pd.DataFrame,
    horizon_hours: int = FORECAST_HORIZON_HOURS,
) -> pd.DataFrame:
    """
    Expand every origin into `horizon_hours` rows (one per hour ahead),
    each containing: origin state (broadcast from the origin), horizon_h,
    TARGET calendar features, climatology, and the actual temperature at
    the target (NaN if it hasn't happened yet -- expected for the
    latest/live origin).

    Args:
        df_state           : Output of run_preprocessing() (one station).
        df_actual           : DataFrame of actual timestamp+temperature
                               (used to look up the y label); may be the
                               same as df_state.
        origins             : Origin timestamps (e.g. from select_origins()).
        climatology_table   : Output of build_climatology().
        horizon_hours       : Hours ahead per origin (default 336 = 14 days).

    Returns:
        DataFrame: origin, horizon_h, timestamp, temperature (y, may be NaN),
        plus feature columns.
    """
    origins = pd.to_datetime(pd.Series(origins).dropna().unique())
    if len(origins) == 0:
        logger.warning("[DirectHorizon] No origins provided — returning empty frame.")
        return pd.DataFrame()

    state_cols   = get_feature_columns(df_state)
    state_lookup = df_state.set_index("timestamp")[state_cols]

    horizons  = np.arange(1, horizon_hours + 1)
    origin_df = pd.DataFrame({"origin": origins})
    horizon_df = pd.DataFrame({"horizon_h": horizons})
    frame = origin_df.merge(horizon_df, how="cross")
    frame["timestamp"] = frame["origin"] + pd.to_timedelta(frame["horizon_h"], unit="h")

    frame = frame.merge(state_lookup, left_on="origin", right_index=True, how="left")
    frame = frame.dropna(subset=state_cols)

    actual_lookup = df_actual.drop_duplicates(subset="timestamp").set_index("timestamp")[TARGET_COL]
    frame[TARGET_COL] = frame["timestamp"].map(actual_lookup)

    target_calendar = build_temporal_features(frame["timestamp"])
    frame = pd.concat([frame.reset_index(drop=True), target_calendar.reset_index(drop=True)], axis=1)

    frame["month"] = frame["timestamp"].dt.month
    frame["day"]   = frame["timestamp"].dt.day
    frame = frame.merge(climatology_table, on=["month", "day"], how="left")
    frame = frame.drop(columns=["month", "day"])

    logger.info(
        f"[DirectHorizon] Built {len(frame):,} rows from {len(origins):,} origins "
        f"x up to {horizon_hours}h horizon."
    )
    return frame


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build dummy data for an end-to-end pipeline sanity check.
    dates = pd.date_range("2023-01-01", periods=24 * 400, freq="h")
    temps = (27 + 3 * np.sin(2 * np.pi * dates.dayofyear / 365)
                + 2 * np.sin(2 * np.pi * dates.hour / 24)
                + np.random.randn(len(dates)) * 0.5)
    dummy = pd.DataFrame({
        "timestamp": dates, "temperature": temps,
        "humidity": 80.0, "wind_speed": 10.0, "precipitation": 0.0,
    })

    processed = run_preprocessing(dummy.copy(), station_name="TestStation")
    print(f"\nOrigin-state rows : {len(processed)}")
    print(f"Origin-state cols : {len(get_feature_columns(processed))}")

    end_train = dates.max() - pd.Timedelta(days=30)
    origins = select_origins(processed, end=end_train, stride_hours=24)
    print(f"Candidate origins (<= {end_train.date()}): {len(origins)}")

    clim  = build_climatology(dummy, as_of=end_train)
    frame = build_direct_horizon_frame(
        processed, dummy, origins.tail(5), clim, horizon_hours=FORECAST_HORIZON_HOURS,
    )
    feat_cols = get_feature_columns(frame)
    print(f"\nDirect-horizon frame: {len(frame):,} rows, {len(feat_cols)} features")
    print(frame[["origin", "horizon_h", "timestamp", "temperature"] + feat_cols[:6]].head(5).to_string())
