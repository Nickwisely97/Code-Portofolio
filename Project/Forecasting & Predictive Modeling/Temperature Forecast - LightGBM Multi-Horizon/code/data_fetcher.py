"""
data_fetcher.py
Fetch temperature data for Indonesian cities from two sources:
  - Open-Meteo Historical API (ERA5 reanalysis) -> long-term hourly history
  - BMKG public forecast API                    -> next few days, per kelurahan

Results are cached per station as `data/<station>/raw.parquet` so repeated
notebook runs don't re-hit the APIs unless FORCE_REFRESH is set.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Station registry: BMKG adm4 (kelurahan) code + coordinates for Open-Meteo.
# The 5 largest Indonesian cities by population.
STATIONS = {
    "Jakarta":  {"adm4": "31.71.01.1001", "lat": -6.2088, "lon": 106.8456},
    "Surabaya": {"adm4": "35.78.01.1001", "lat": -7.2575, "lon": 112.7521},
    "Bandung":  {"adm4": "32.73.01.1001", "lat": -6.9175, "lon": 107.6191},
    "Medan":    {"adm4": "12.71.01.1001", "lat": 3.5952, "lon": 98.6722},  # Medan is north of the equator
    "Semarang": {"adm4": "33.74.01.1001", "lat": -6.9932, "lon": 110.4203},
}

BMKG_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = "data"

RAW_COLUMNS = ["timestamp", "temperature", "humidity", "wind_speed",
               "precipitation", "station", "source", "weather_code"]


def fetch_bmkg_forecast(name: str, adm4: str) -> pd.DataFrame:
    """Fetch a few days of forecast from BMKG for one kelurahan code."""
    try:
        resp = requests.get(f"{BMKG_URL}?adm4={adm4}", timeout=10)
        resp.raise_for_status()
        cuaca = [item for day in resp.json().get("data", [{}])[0].get("cuaca", []) for item in day]

        df = pd.DataFrame(cuaca)
        if df.empty:
            return pd.DataFrame(columns=RAW_COLUMNS)

        rename_map = {
            "local_datetime": "timestamp",
            "t":  "temperature",
            "hu": "humidity",
            "ws": "wind_speed",
            "tp": "precipitation",
            "weather": "weather_code",
        }
        df = df.rename(columns=rename_map)
        for col in ["temperature", "humidity", "wind_speed", "precipitation", "weather_code"]:
            if col not in df.columns:
                df[col] = np.nan

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        num_cols = ["temperature", "humidity", "wind_speed", "precipitation", "weather_code"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric)
        df["station"], df["source"] = name, "bmkg_forecast"
        return df[RAW_COLUMNS]
    except Exception as e:
        logger.error(f"Failed to fetch BMKG forecast for {name}: {e}")
        return pd.DataFrame(columns=RAW_COLUMNS)


def fetch_historical(name: str, lat: float, lon: float, start: str = "2019-01-01") -> pd.DataFrame:
    """Fetch hourly ERA5 reanalysis history from Open-Meteo."""
    end = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "Asia/Jakarta",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})

        df = pd.DataFrame({
            "timestamp":     pd.to_datetime(h.get("time")),
            "temperature":   h.get("temperature_2m"),
            "humidity":      h.get("relative_humidity_2m"),
            "wind_speed":    h.get("wind_speed_10m"),
            "precipitation": h.get("precipitation"),
            "station":       name,
            "source":        "openmeteo_historical",
            "weather_code":  np.nan,
        })
        return df[RAW_COLUMNS]
    except Exception as e:
        logger.error(f"Failed to fetch Open-Meteo historical data for {name}: {e}")
        return pd.DataFrame(columns=RAW_COLUMNS)


def load_station_data(name: str) -> pd.DataFrame:
    """Load a station's cached raw parquet, if present."""
    path = os.path.join(DATA_DIR, name.lower(), "raw.parquet")
    if os.path.exists(path):
        logger.info(f"[Cache] Loading {name} from {path}")
        return pd.read_parquet(path)
    return pd.DataFrame(columns=RAW_COLUMNS)


def fetch_all_stations(
    stations: dict,
    historical_start: str = "2019-01-01",
    force_refresh: bool = False,
) -> dict:
    """
    Fetch (or load from cache) raw data for each station in `stations`.

    Args:
        stations         : dict {name: {"adm4":..., "lat":..., "lon":...}}
        historical_start : start date for the Open-Meteo history pull
        force_refresh     : ignore cache and re-fetch from both APIs

    Returns:
        dict {name: DataFrame} with columns RAW_COLUMNS.
    """
    result = {}

    for name, cfg in stations.items():
        station_dir = os.path.join(DATA_DIR, name.lower())
        os.makedirs(station_dir, exist_ok=True)
        path = os.path.join(station_dir, "raw.parquet")

        if not force_refresh and os.path.exists(path):
            logger.info(f"[Cache] Loading {name} from {path}")
            result[name] = pd.read_parquet(path)
            continue

        logger.info(f"[Fetch] {name}: fetching historical data (Open-Meteo) + forecast (BMKG) ...")
        df_hist = fetch_historical(name, cfg["lat"], cfg["lon"], start=historical_start)
        df_fore = fetch_bmkg_forecast(name, cfg["adm4"])

        df_all = pd.concat([df_hist, df_fore], ignore_index=True)
        if not df_all.empty:
            df_all = (df_all.drop_duplicates(subset="timestamp")
                             .sort_values("timestamp")
                             .reset_index(drop=True))
            df_all.to_parquet(path, index=False)

        result[name] = df_all
        time.sleep(1)  # rate limit safety between stations

    return result


if __name__ == "__main__":
    data = fetch_all_stations(STATIONS, force_refresh=True)
    for name, df in data.items():
        print(f"{name}: {len(df)} rows")
