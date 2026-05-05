import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfigurasi Lokasi
STATIONS = {
    "Jakarta":   {"adm4": "31.71.01.1001", "lat": -6.2088, "lon": 106.8456},
    "Surabaya":  {"adm4": "35.78.01.1001", "lat": -7.2575, "lon": 112.7521},
    "Bandung":   {"adm4": "32.73.01.1001", "lat": -6.9175, "lon": 107.6191},
}

BMKG_URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = "data"

def fetch_bmkg_forecast(name, adm4):
    """Ambil forecast BMKG dengan struktur yang lebih flat."""
    try:
        resp = requests.get(f"{BMKG_URL}?adm4={adm4}", timeout=10)
        resp.raise_for_status()
        # Mengambil semua jam dari semua grup hari sekaligus menggunakan list comprehension
        cuaca = [item for day in resp.json().get("data", [{}])[0].get("cuaca", []) for item in day]
        
        df = pd.DataFrame(cuaca)
        if df.empty: return df

        # Mapping kolom BMKG ke standar kita
        rename_map = {"local_datetime": "timestamp", "t": "temperature", "hu": "humidity", "ws": "wind_speed"}
        df = df.rename(columns=rename_map)[list(rename_map.values())]
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df[["temperature", "humidity", "wind_speed"]] = df[["temperature", "humidity", "wind_speed"]].apply(pd.to_numeric)
        df["station"], df["source"] = name, "bmkg"
        return df
    except Exception as e:
        logger.error(f"Gagal BMKG {name}: {e}")
        return pd.DataFrame()

def fetch_historical(name, lat, lon, start="2019-01-01"):
    """Ambil data historis Open-Meteo."""
    end = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Jakarta"
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        h = resp.json().get("hourly", {})
        
        return pd.DataFrame({
            "timestamp": pd.to_datetime(h.get("time")),
            "temperature": h.get("temperature_2m"),
            "humidity": h.get("relative_humidity_2m"),
            "wind_speed": h.get("wind_speed_10m"),
            "station": name, "source": "openmeteo"
        })
    except Exception as e:
        logger.error(f"Gagal Historical {name}: {e}")
        return pd.DataFrame()

def sync_data(force=False):
    """Pipeline utama untuk semua stasiun."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, cfg in STATIONS.items():
        path = os.path.join(DATA_DIR, f"{name.lower()}.parquet")
        if os.path.exists(path) and not force: continue

        logger.info(f"Proses stasiun: {name}")
        df_h = fetch_historical(name, cfg['lat'], cfg['lon'])
        df_f = fetch_bmkg_forecast(name, cfg['adm4'])
        
        full_df = pd.concat([df_h, df_f]).drop_duplicates("timestamp").sort_values("timestamp")
        full_df.to_parquet(path, index=False)
        time.sleep(1) # Rate limit safety

if __name__ == "__main__":
    sync_data(force=True)