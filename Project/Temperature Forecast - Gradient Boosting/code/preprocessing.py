"""
preprocessing.py
Feature engineering untuk prediksi temperatur Indonesia.

Features yang dibangun:
  • Temporal cyclical  : sin/cos jam, hari-dalam-minggu, hari-dalam-bulan, bulan, hari-dalam-tahun
  • Kalender           : is_weekend, quarter, year
  • Lag features       : suhu pada t-1, t-2, t-3, t-6, t-12, t-24, t-48 jam
  • Rolling stats      : mean & std untuk window 7, 14, 30, 60, 90 hari
  • Momentum          : selisih suhu antar periode (1, 3, 6, 12, 24 jam)
  • Humidity & wind   : fitur cuaca tambahan (jika tersedia)
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── Konstanta ──────────────────────────────────────────────────────────────
LAG_HOURS          = [1, 2, 3, 6, 12, 24, 48, 72]
ROLLING_DAYS       = [7, 14, 30, 60, 90]          # dalam hari → ×24 jam
MOMENTUM_HOURS     = [1, 3, 6, 12, 24]
TARGET_COL         = "temperature"


# ─── Helper: Cyclical Encoding ───────────────────────────────────────────────
def _cyclic(series: pd.Series, max_val: float):
    """Encode series sebagai (sin, cos) pair untuk representasi siklus."""
    angle = 2 * np.pi * series / max_val
    return np.sin(angle), np.cos(angle)


# ─── Feature Engineering ─────────────────────────────────────────────────────
def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan fitur waktu siklus dari kolom 'timestamp'.
    Sin/cos encoding untuk: jam, hari_minggu, hari_bulan, bulan, hari_tahun.
    """
    df = df.copy()
    ts = df["timestamp"]

    # --- jam (0–23) ---
    sin_h, cos_h = _cyclic(ts.dt.hour, 24)
    df["sin_hour"], df["cos_hour"] = sin_h, cos_h

    # --- hari dalam minggu (0=Senin … 6=Minggu) ---
    sin_dow, cos_dow = _cyclic(ts.dt.dayofweek, 7)
    df["sin_dayofweek"], df["cos_dayofweek"] = sin_dow, cos_dow

    # --- hari dalam bulan (1–31) ---
    sin_dom, cos_dom = _cyclic(ts.dt.day, 31)
    df["sin_dayofmonth"], df["cos_dayofmonth"] = sin_dom, cos_dom

    # --- bulan (1–12) ---
    sin_m, cos_m = _cyclic(ts.dt.month, 12)
    df["sin_month"], df["cos_month"] = sin_m, cos_m

    # --- hari dalam tahun (1–365/366) ---
    sin_doy, cos_doy = _cyclic(ts.dt.dayofyear, 366)
    df["sin_dayofyear"], df["cos_dayofyear"] = sin_doy, cos_doy

    # --- fitur kalender sederhana ---
    df["hour"]       = ts.dt.hour.astype(np.int8)
    df["dayofweek"]  = ts.dt.dayofweek.astype(np.int8)
    df["month"]      = ts.dt.month.astype(np.int8)
    df["year"]       = ts.dt.year.astype(np.int16)
    df["quarter"]    = ts.dt.quarter.astype(np.int8)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)

    # --- indikator musim hujan Indonesia (Nov–Mar) ---
    df["is_rainy_season"] = ts.dt.month.isin([11, 12, 1, 2, 3]).astype(np.int8)

    logger.debug("Temporal features selesai.")
    return df


def build_lag_features(df: pd.DataFrame, lags: List[int] = LAG_HOURS) -> pd.DataFrame:
    """
    Lag features untuk TARGET_COL.
    df harus sudah diurutkan berdasarkan timestamp dan memiliki frekuensi jam yang konsisten.
    """
    df = df.copy()
    for lag in lags:
        col_name         = f"lag_{lag}h"
        df[col_name]     = df[TARGET_COL].shift(lag)
    logger.debug(f"Lag features ({lags}) selesai.")
    return df


def build_rolling_features(df: pd.DataFrame, windows_days: List[int] = ROLLING_DAYS) -> pd.DataFrame:
    """
    Rolling mean dan std untuk TARGET_COL.
    Window dinyatakan dalam hari, dikonversi ke jam (×24).
    min_periods = 50% dari ukuran window agar tidak terlalu banyak NaN di awal.
    """
    df = df.copy()
    for days in windows_days:
        w           = days * 24          # jam
        min_p       = max(1, w // 2)
        col_mean    = f"roll_mean_{days}d"
        col_std     = f"roll_std_{days}d"
        col_min     = f"roll_min_{days}d"
        col_max     = f"roll_max_{days}d"
        rolled      = df[TARGET_COL].shift(1).rolling(window=w, min_periods=min_p)
        df[col_mean] = rolled.mean()
        df[col_std]  = rolled.std()
        df[col_min]  = rolled.min()
        df[col_max]  = rolled.max()
    logger.debug(f"Rolling features ({windows_days} days) selesai.")
    return df


def build_momentum_features(df: pd.DataFrame, periods: List[int] = MOMENTUM_HOURS) -> pd.DataFrame:
    """
    Momentum = selisih suhu dengan t-n jam yang lalu.
    Mengukur arah dan kecepatan perubahan suhu.
    """
    df = df.copy()
    for p in periods:
        df[f"momentum_{p}h"] = df[TARGET_COL].diff(p)
    logger.debug(f"Momentum features ({periods}) selesai.")
    return df


def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur tambahan dari kolom cuaca (jika tersedia): humidity, wind_speed, precipitation.
    Termasuk lag-1 dan lag-24 untuk setiap kolom yang ada.
    """
    df = df.copy()
    aux_cols = ["humidity", "wind_speed", "precipitation"]

    for col in aux_cols:
        if col in df.columns:
            # Isi missing values dengan forward fill
            df[col] = df[col].ffill().bfill()
            df[f"{col}_lag1"]  = df[col].shift(1)
            df[f"{col}_lag24"] = df[col].shift(24)

    # Interaksi suhu × kelembapan (heat index proxy)
    if "humidity" in df.columns:
        df["temp_humidity_interact"] = df[TARGET_COL] * df["humidity"] / 100

    logger.debug("Weather auxiliary features selesai.")
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

    # Interpolasi suhu (kolom numerik utama)
    if TARGET_COL in df_hourly.columns:
        df_hourly[TARGET_COL] = df_hourly[TARGET_COL].interpolate(method="time", limit=6)

    # Forward-fill kolom lain (max 6 jam)
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
    Pipeline lengkap preprocessing.

    Args:
        df           : DataFrame raw dengan kolom 'timestamp' dan 'temperature'.
        station_name : Nama stasiun (opsional, untuk logging).
        drop_na      : Buang baris yang masih NaN setelah feature engineering.

    Returns:
        DataFrame dengan semua features.
    """
    label = station_name or "unknown"
    logger.info(f"[Preprocessing] Mulai untuk stasiun: {label} | baris awal: {len(df)}")

    # 0. Pastikan timestamp adalah datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Resample ke frekuensi jam yang konsisten
    df = resample_to_hourly(df)

    # 2. Fitur temporal (sin/cos seasonality)
    df = build_temporal_features(df)

    # 3. Lag features
    df = build_lag_features(df)

    # 4. Rolling features
    df = build_rolling_features(df)

    # 5. Momentum features
    df = build_momentum_features(df)

    # 6. Fitur cuaca tambahan
    df = build_weather_features(df)

    # 7. Tambahkan kolom station (jika belum ada)
    if station_name and "station" not in df.columns:
        df["station"] = station_name

    # 8. Buang NaN (baris pertama akan banyak NaN akibat lag/rolling besar)
    rows_before = len(df)
    if drop_na:
        # Hanya drop baris yang NaN di kolom target
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    rows_after = len(df)
    logger.info(f"[Preprocessing] Selesai: {rows_after}/{rows_before} baris valid.")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Kembalikan daftar kolom fitur (exclude: timestamp, station, source, target).
    """
    exclude = {"timestamp", "station", "source", TARGET_COL, "weather_code"}
    return [c for c in df.columns if c not in exclude]


def split_train_test(
    df: pd.DataFrame,
    end_train: str,
    target_date: str,
) -> tuple:
    """
    Pisahkan data menjadi train dan test berdasarkan tanggal.

    Args:
        df          : DataFrame hasil preprocessing (sudah ada timestamp).
        end_train   : Tanggal akhir data training (inclusive), format 'YYYY-MM-DD'.
        target_date : Tanggal akhir data test / prediksi, format 'YYYY-MM-DD'.

    Returns:
        (df_train, df_test) — tuple dua DataFrame.
    """
    end_train_dt   = pd.to_datetime(end_train)
    target_date_dt = pd.to_datetime(target_date) + pd.Timedelta(hours=23)

    df_train = df[df["timestamp"] <= end_train_dt].copy()
    df_test  = df[(df["timestamp"] > end_train_dt) & (df["timestamp"] <= target_date_dt)].copy()

    logger.info(
        f"Train: {len(df_train)} baris ({df_train['timestamp'].min()} → {df_train['timestamp'].max()})\n"
        f"Test : {len(df_test)} baris ({df_test['timestamp'].min()} → {df_test['timestamp'].max()})"
    )
    return df_train, df_test


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Buat data dummy untuk verifikasi pipeline
    dates  = pd.date_range("2023-01-01", periods=24 * 100, freq="h")
    temps  = 27 + 3 * np.sin(2 * np.pi * dates.hour / 24) + np.random.randn(len(dates)) * 0.5
    dummy  = pd.DataFrame({"timestamp": dates, "temperature": temps, "humidity": 80.0, "wind_speed": 10.0})

    processed = run_preprocessing(dummy, station_name="TestStation")
    feat_cols  = get_feature_columns(processed)

    print(f"\nTotal fitur : {len(feat_cols)}")
    print(f"Baris valid : {len(processed)}")
    print("\nSampel fitur:")
    print(processed[["timestamp", "temperature"] + feat_cols[:10]].tail(3).to_string())
