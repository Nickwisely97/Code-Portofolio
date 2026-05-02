"""
data_fetcher.py
Mengambil data dari:
1. BMKG API - Data prakiraan cuaca (forecast) terkini per kelurahan
2. Open-Meteo Historical API - Data historis suhu (ERA5 reanalysis) sebagai training data
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Konfigurasi Stasiun / Lokasi ────────────────────────────────────────────
STATIONS = {
    "Jakarta":       {"adm4": "31.71.01.1001", "lat": -6.2088, "lon": 106.8456},
    "Surabaya":      {"adm4": "35.78.01.1001", "lat": -7.2575, "lon": 112.7521},
    "Bandung":       {"adm4": "32.73.01.1001", "lat": -6.9175, "lon": 107.6191},
    "Medan":         {"adm4": "12.71.01.1001", "lat":  3.5896, "lon":  98.6731},
    "Makassar":      {"adm4": "73.71.01.1001", "lat": -5.1477, "lon": 119.4327},
    "Yogyakarta":    {"adm4": "34.71.01.1001", "lat": -7.7956, "lon": 110.3695},
    "Denpasar":      {"adm4": "51.71.01.1001", "lat": -8.6705, "lon": 115.2126},
    "Palembang":     {"adm4": "16.71.01.1001", "lat": -2.9761, "lon": 104.7754},
}

BMKG_BASE_URL    = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
OPENMETEO_URL    = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR         = "data"
OUTPUT_DIR       = "output"


def fetch_bmkg_forecast(station_name: str, station_cfg: dict) -> pd.DataFrame:
    """
    Tarik data prakiraan cuaca BMKG per kelurahan (3-hari).
    Mengembalikan DataFrame dengan kolom: timestamp, temperature, humidity, wind_speed, station.
    """
    url    = f"{BMKG_BASE_URL}?adm4={station_cfg['adm4']}&receive=timestamp,temperature,humidity,wind_speed"
    logger.info(f"[BMKG] Fetching forecast for {station_name} ...")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.warning(f"[BMKG] Gagal mengambil data {station_name}: {e}")
        return pd.DataFrame()

    records = []
    cuaca_list = raw.get("data", [{}])[0].get("cuaca", [])

    for day_group in cuaca_list:
        for item in day_group:
            try:
                records.append({
                    "timestamp":    pd.to_datetime(item.get("local_datetime") or item.get("datetime")),
                    "temperature":  float(item.get("t", np.nan)),
                    "humidity":     float(item.get("hu", np.nan)),
                    "wind_speed":   float(item.get("ws", np.nan)),
                    "weather_code": item.get("weather", np.nan),
                    "station":      station_name,
                    "source":       "bmkg_forecast",
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        logger.info(f"[BMKG] {station_name}: {len(df)} records diperoleh.")
    return df


def fetch_openmeteo_historical(
    station_name: str,
    lat: float,
    lon: float,
    start_date: str = "2019-01-01",
    end_date:   str = None,
) -> pd.DataFrame:
    """
    Tarik data historis suhu per-jam dari Open-Meteo (ERA5 reanalysis).
    Data sejak 2019 agar model punya konteks musiman yang panjang.
    """
    if end_date is None:
        end_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "latitude":              lat,
        "longitude":             lon,
        "start_date":            start_date,
        "end_date":              end_date,
        "hourly":                "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone":              "Asia/Jakarta",
        "wind_speed_unit":       "kmh",
    }

    logger.info(f"[Open-Meteo] Fetching historical data for {station_name} ({start_date} → {end_date}) ...")

    try:
        resp = requests.get(OPENMETEO_URL, params=params, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error(f"[Open-Meteo] Gagal: {e}")
        return pd.DataFrame()

    hourly = raw.get("hourly", {})
    df = pd.DataFrame({
        "timestamp":   pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity":    hourly["relative_humidity_2m"],
        "wind_speed":  hourly["wind_speed_10m"],
        "precipitation": hourly.get("precipitation", [np.nan] * len(hourly["time"])),
        "station":     station_name,
        "source":      "openmeteo_historical",
    })

    df = df.dropna(subset=["temperature"]).sort_values("timestamp").drop_duplicates("timestamp")
    logger.info(f"[Open-Meteo] {station_name}: {len(df)} records diperoleh.")
    return df


def fetch_all_stations(
    stations: dict = None,
    historical_start: str = "2019-01-01",
    force_refresh: bool = False,
) -> dict:
    """
    Pipeline utama: ambil data historis (Open-Meteo) + forecast terkini (BMKG)
    untuk semua stasiun. Cache hasil ke disk agar tidak perlu re-fetch.
    
    Returns:
        dict  {station_name: DataFrame}
    """
    if stations is None:
        stations = STATIONS

    os.makedirs(DATA_DIR, exist_ok=True)
    result = {}

    for name, cfg in stations.items():
        cache_path = os.path.join(DATA_DIR, f"{name.lower()}_raw.parquet")

        if os.path.exists(cache_path) and not force_refresh:
            logger.info(f"[Cache] Loading {name} dari {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # 1) Ambil historis (panjang)
            df_hist = fetch_openmeteo_historical(
                name, cfg["lat"], cfg["lon"],
                start_date=historical_start,
            )
            # 2) Ambil forecast BMKG (terkini)
            df_fore = fetch_bmkg_forecast(name, cfg)

            # 3) Gabungkan dan deduplicate
            df = pd.concat([df_hist, df_fore], ignore_index=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

            df.to_parquet(cache_path, index=False)
            logger.info(f"[Saved] {name} → {cache_path} ({len(df)} rows)")

        result[name] = df
        time.sleep(0.5)   # jaga rate-limit Open-Meteo

    return result


def load_station_data(station_name: str) -> pd.DataFrame:
    """Load data stasiun dari cache (parquet)."""
    path = os.path.join(DATA_DIR, f"{station_name.lower()}_raw.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache tidak ditemukan: {path}. Jalankan fetch_all_stations() dulu.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    data = fetch_all_stations(force_refresh=True)
    for name, df in data.items():
        print(f"\n{'='*50}")
        print(f"Stasiun : {name}")
        print(f"Baris   : {len(df)}")
        print(f"Periode : {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(df[["timestamp", "temperature", "humidity", "wind_speed"]].tail(3))
