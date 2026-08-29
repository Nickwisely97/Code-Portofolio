"""
preprocessing.py
Feature engineering untuk prediksi temperatur Indonesia.

Pipeline dibangun di sekitar konsep forecast origin T0: model hanya boleh
memakai informasi yang diketahui pada/sebelum T0, lalu memprediksi setiap jam
dari T0+1h sampai T0+horizon_hours (default 336 jam / 14 hari) sekaligus
(direct multi-horizon), bukan autoregresif per jam.

Dua kelompok fitur:
  1. Origin state  — dihitung sekali per origin dari data <= T0:
       lag suhu (t-1, t-3, t-6, t-12, t-24, t-48, t-72 jam)
       rolling stats suhu (mean/std/min/max, window 7/14/30/60/90 hari)
       momentum suhu (selisih t-6h, t-24h)
       humidity/wind_speed/precipitation saat origin + lag-1/lag-24 jam
     Nilai-nilai ini di-broadcast ke seluruh baris horizon origin tsb.
  2. Target calendar — dihitung dari timestamp TARGET (T0+h), bukan T0,
     karena fakta kalender (jam, hari, bulan, musim) untuk tanggal manapun
     di masa depan selalu bisa diketahui lebih dulu, jadi aman dipakai.

Plus `horizon_h` (jam ke depan dari origin) dan climatology (rata-rata/std
historis suhu per (bulan, tanggal), dihitung hanya dari data <= suatu
cutoff) sebagai sinyal tambahan untuk horizon yang jauh.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Konstanta ──────────────────────────────────────────────────────────────
LAG_HOURS              = [1, 3, 6, 12, 24, 48, 72]
ROLLING_DAYS           = [7, 14, 30, 60, 90]          # dalam hari → ×24 jam
MOMENTUM_HOURS         = [6, 24]
WEATHER_AUX_COLS       = ["humidity", "wind_speed", "precipitation"]
TARGET_COL             = "temperature"
ORIGIN_STRIDE_HOURS    = 24     # jarak antar forecast origin saat sampling
FORECAST_HORIZON_HOURS = 336    # 14 hari


# ─── Helper: Cyclical Encoding ───────────────────────────────────────────────
def _cyclic(series: pd.Series, max_val: float):
    """Encode series sebagai (sin, cos) pair untuk representasi siklus."""
    angle = 2 * np.pi * series / max_val
    return np.sin(angle), np.cos(angle)


# ─── Target-side calendar features ───────────────────────────────────────────
def build_temporal_features(ts: pd.Series) -> pd.DataFrame:
    """
    Fitur kalender siklus untuk sebuah Series timestamp (dipakai pada
    timestamp TARGET saat membangun direct-horizon frame — bukan pada
    origin — karena fakta kalender tanggal manapun selalu bisa diketahui
    lebih dulu).

    Args:
        ts : Series datetime.

    Returns:
        DataFrame kolom-kolom kalender, index sama dengan `ts`.
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
    Lag suhu relatif terhadap timestamp masing-masing baris. Dipakai sebagai
    origin state: baris pada T0 merepresentasikan suhu yang sudah diketahui
    N jam sebelum T0.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[TARGET_COL].shift(lag)
    logger.debug(f"Lag features ({lags}) selesai.")
    return df


def build_rolling_features(df: pd.DataFrame, windows_days: List[int] = ROLLING_DAYS) -> pd.DataFrame:
    """
    Rolling mean/std/min/max suhu, dihitung dari data SEBELUM timestamp
    masing-masing baris (shift(1) dulu) — origin state, bukan future info.
    min_periods = 50% dari ukuran window agar tidak terlalu banyak NaN di awal.
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
    logger.debug(f"Rolling features ({windows_days} days) selesai.")
    return df


def build_momentum_features(df: pd.DataFrame, periods: List[int] = MOMENTUM_HOURS) -> pd.DataFrame:
    """Momentum = selisih suhu dengan t-n jam yang lalu (arah & kecepatan perubahan)."""
    df = df.copy()
    for p in periods:
        df[f"momentum_{p}h"] = df[TARGET_COL].diff(p)
    logger.debug(f"Momentum features ({periods}) selesai.")
    return df


def build_weather_features(df: pd.DataFrame, aux_cols: List[str] = WEATHER_AUX_COLS) -> pd.DataFrame:
    """
    Fitur cuaca tambahan sebagai origin state: nilai humidity/wind_speed/
    precipitation PADA origin (`{col}_at_origin`) plus lag-1 dan lag-24 jam.
    Kolom mentah (kontemporer terhadap baris) DIBUANG dari hasil akhir —
    untuk horizon lebih dari beberapa jam, nilai cuaca "saat ini" tidak
    diketahui, hanya nilai historis sampai origin yang boleh dipakai.
    """
    df = df.copy()
    present = [c for c in aux_cols if c in df.columns]

    for col in present:
        filled = df[col].ffill().bfill()
        df[f"{col}_at_origin"] = filled
        df[f"{col}_lag1"]      = filled.shift(1)
        df[f"{col}_lag24"]     = filled.shift(24)

    df = df.drop(columns=present)
    logger.debug("Weather auxiliary (origin-state) features selesai.")
    return df


def build_origin_state(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan seluruh origin-state builder di atas. Baris hasilnya, pada
    timestamp T0, merepresentasikan "apa yang sudah diketahui pada T0" —
    dipakai berulang (broadcast) untuk semua horizon origin T0 tersebut,
    bukan sebagai fitur milik baris T0 itu sendiri.
    """
    df = build_lag_features(df_hourly)
    df = build_rolling_features(df)
    df = build_momentum_features(df)
    df = build_weather_features(df)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan DataFrame memiliki frekuensi jam yang reguler (1H).
    Gap akan diisi dengan interpolasi linear untuk suhu, forward-fill untuk yang lain.
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
    Bangun tabel origin-state lengkap dari data mentah.

    Args:
        df           : DataFrame raw dengan kolom 'timestamp' dan 'temperature'.
        station_name : Nama stasiun (opsional, untuk logging & kolom 'station').
        drop_na      : Buang baris yang state-nya masih NaN (terlalu awal di
                        histori untuk lag/rolling besar) — baris ini tidak
                        bisa dipakai sebagai origin.

    Returns:
        DataFrame origin-state: timestamp, temperature, [station], + fitur
        lag/rolling/momentum/cuaca. (Fitur kalender TIDAK ada di sini —
        itu dihitung belakangan untuk timestamp target, lihat
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
    Kembalikan daftar kolom fitur, exclude kolom bookkeeping/target:
    timestamp, station, source, origin, temperature, weather_code.
    Dipakai baik untuk tabel origin-state maupun direct-horizon frame.
    """
    exclude = {"timestamp", "station", "source", "origin", TARGET_COL, "weather_code"}
    return [c for c in df.columns if c not in exclude]


# ─── Climatology ──────────────────────────────────────────────────────────────
def build_climatology(df_actual: pd.DataFrame, as_of) -> pd.DataFrame:
    """
    Rata-rata & std suhu historis per (bulan, tanggal), dihitung HANYA dari
    data pada/sebelum `as_of` — supaya tetap aman dari leakage saat dipakai
    sebagai fitur untuk origin manapun >= as_of.

    Returns:
        DataFrame kolom: month, day, climatology_mean, climatology_std.
    """
    as_of = pd.to_datetime(as_of)
    hist = df_actual[df_actual["timestamp"] <= as_of].copy()
    hist["month"] = hist["timestamp"].dt.month
    hist["day"]   = hist["timestamp"].dt.day

    clim = (hist.groupby(["month", "day"])[TARGET_COL]
                .agg(climatology_mean="mean", climatology_std="std")
                .reset_index())

    # Fallback untuk (bulan, tanggal) dengan sample terlalu sedikit (std NaN).
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
    Pilih timestamp yang layak dipakai sebagai forecast origin: state-nya
    lengkap (tidak NaN), berada dalam rentang [start, end], dan berjarak
    `stride_hours` satu sama lain (default: harian) supaya jumlah origin
    tetap terkendali saat dipakai membangun training/backtest set.
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
    Ekspansi setiap origin menjadi `horizon_hours` baris (satu per jam ke
    depan), masing-masing berisi: origin state (broadcast dari origin),
    horizon_h, fitur kalender TARGET, climatology, dan suhu aktual pada
    target (NaN kalau belum benar-benar terjadi — wajar untuk origin
    terbaru/live).

    Args:
        df_state          : Hasil run_preprocessing() (satu stasiun).
        df_actual         : DataFrame timestamp+temperature aktual (dipakai
                             untuk lookup label y); boleh sama dengan df_state.
        origins            : Timestamp-timestamp origin (mis. dari select_origins()).
        climatology_table  : Hasil build_climatology().
        horizon_hours       : Berapa jam ke depan per origin (default 336 = 14 hari).

    Returns:
        DataFrame: origin, horizon_h, timestamp, temperature (y, bisa NaN),
        + kolom fitur.
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
    # Buat data dummy untuk verifikasi pipeline end-to-end.
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
