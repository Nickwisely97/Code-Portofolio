"""
main_bmkg_forecast.py
─────────────────────
Orchestrator utama pipeline prediksi temperatur Indonesia menggunakan data BMKG
dan Open-Meteo.

Pipeline:
  1. Tarik data (BMKG forecast + Open-Meteo historical)
  2. Preprocessing & feature engineering
  3. Split train / test berdasarkan END_TRAIN & TARGET_DATE
  4. Latih model LightGBM
  5. Prediksi & evaluasi
  6. Plot hasil & simpan output
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict

# ── Import modul lokal ─────────────────────────────────────────────────────
from data_fetcher   import fetch_all_stations, STATIONS, load_station_data
from preprocessing  import run_preprocessing, TARGET_COL
from model          import TemperatureForecastModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konfigurasi default ───────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Tanggal akhir data training (model belajar sampai tanggal ini)
    "END_TRAIN":          "2024-12-31",

    # Tanggal akhir prediksi yang diinginkan
    "TARGET_DATE":        "2025-04-30",

    # Stasiun yang dijalankan (None = semua)
    "STATIONS":           ["Jakarta", "Surabaya", "Bandung"],

    # Mulai tarik data historis dari tanggal ini
    "HISTORICAL_START":   "2019-01-01",

    # Paksa re-fetch data meski cache sudah ada
    "FORCE_REFRESH":      False,

    # Simpan model setelah training
    "SAVE_MODEL":         True,

    # Tampilkan grafik interaktif (set True jika ada display)
    "SHOW_PLOTS":         False,

    # Direktori output
    "OUTPUT_DIR":         "../output",

    # Hyperparameter LightGBM (override jika diperlukan)
    "LGBM_PARAMS":        None,
}


# ─────────────────────────────────────────────────────────────────────────────
def validate_dates(end_train: str, target_date: str):
    """Validasi urutan tanggal."""
    et = pd.to_datetime(end_train)
    td = pd.to_datetime(target_date)
    if td <= et:
        raise ValueError(
            f"TARGET_DATE ({target_date}) harus setelah END_TRAIN ({end_train})."
        )
    logger.info(f"Periode: Training s/d {end_train}  |  Prediksi s/d {target_date}")


# ─────────────────────────────────────────────────────────────────────────────
def step_fetch_data(
    station_names:     list,
    historical_start:  str,
    force_refresh:     bool,
) -> Dict[str, pd.DataFrame]:
    """
    STEP 1 — Tarik data dari BMKG & Open-Meteo.
    Mengembalikan dict {station_name: DataFrame raw}.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Fetch Data")
    logger.info("=" * 60)

    target_stations = {k: v for k, v in STATIONS.items() if k in station_names}
    if not target_stations:
        raise ValueError(f"Tidak ada stasiun valid. Pilihan: {list(STATIONS.keys())}")

    data = fetch_all_stations(
        stations        = target_stations,
        historical_start= historical_start,
        force_refresh   = force_refresh,
    )
    for name, df in data.items():
        if df.empty:
            logger.warning(f"  ⚠  Data {name} kosong!")
        else:
            logger.info(
                f"  ✓  {name}: {len(df):,} baris | "
                f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
            )
    return data


# ─────────────────────────────────────────────────────────────────────────────
def step_preprocess(
    raw_data:     Dict[str, pd.DataFrame],
    end_train:    str,
    target_date:  str,
) -> Dict[str, pd.DataFrame]:
    """
    STEP 2 — Preprocessing & feature engineering per stasiun.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    processed = {}
    for name, df_raw in raw_data.items():
        if df_raw.empty:
            logger.warning(f"  ⚠  {name}: data kosong, dilewati.")
            continue
        try:
            df_proc = run_preprocessing(df_raw, station_name=name, drop_na=True)
            # Pastikan data cukup panjang
            n_train = (df_proc["timestamp"] <= pd.to_datetime(end_train)).sum()
            if n_train < 500:
                logger.warning(
                    f"  ⚠  {name}: data training hanya {n_train} baris — mungkin tidak cukup."
                )
            processed[name] = df_proc
            logger.info(
                f"  ✓  {name}: {len(df_proc):,} baris | "
                f"{len([c for c in df_proc.columns if c not in ('timestamp','station','source',TARGET_COL)])} fitur"
            )
        except Exception as e:
            logger.error(f"  ✗  {name}: gagal preprocessing — {e}")

    return processed


# ─────────────────────────────────────────────────────────────────────────────
def step_train_predict(
    processed_data: Dict[str, pd.DataFrame],
    end_train:      str,
    target_date:    str,
    save_model:     bool,
    show_plots:     bool,
    lgbm_params:    Optional[dict],
    output_dir:     str,
) -> Dict[str, dict]:
    """
    STEP 3 & 4 — Training, prediksi, evaluasi, dan plotting per stasiun.
    """
    logger.info("=" * 60)
    logger.info("STEP 3 & 4: Training, Prediksi, Evaluasi, Plot")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, df_proc in processed_data.items():
        logger.info(f"\n── Stasiun: {name} ──────────────────────────────────")
        try:
            # Inisialisasi model
            mdl = TemperatureForecastModel(
                end_train   = end_train,
                target_date = target_date,
                station     = name,
                lgbm_params = lgbm_params,
            )

            # Training
            logger.info(f"  ▶  Training model LightGBM …")
            mdl.fit(df_proc)

            # Prediksi test window
            df_test_result = mdl.predict()

            # Prediksi full (untuk grafik)
            df_full_result = mdl.predict_full(df_proc)

            # Metrics
            metrics = mdl.get_metrics()
            logger.info(
                f"  📊  MAE={metrics.get('mae', 0):.3f}°C | "
                f"RMSE={metrics.get('rmse', 0):.3f}°C | "
                f"MAPE={metrics.get('mape', 0):.2f}% | "
                f"R²={metrics.get('r2', 0):.4f}"
            )

            # Simpan prediksi ke CSV
            csv_path = os.path.join(output_dir, f"{name.lower()}_predictions.csv")
            df_test_result.to_csv(csv_path, index=False)
            logger.info(f"  💾  Prediksi disimpan → {csv_path}")

            # Plot
            img_main    = mdl.plot_results(df_full_result, last_n_days=120, show=show_plots)
            img_horizon = mdl.plot_forecast_horizon(df_full_result, horizon_days=7, show=show_plots)

            # Simpan model
            model_path = ""
            if save_model:
                model_path = mdl.save_model()

            results[name] = {
                "model":        mdl,
                "metrics":      metrics,
                "predictions":  df_test_result,
                "full_preds":   df_full_result,
                "csv_path":     csv_path,
                "img_main":     img_main,
                "img_horizon":  img_horizon,
                "model_path":   model_path,
            }

        except Exception as e:
            logger.error(f"  ✗  {name}: gagal — {e}", exc_info=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
def step_summary(results: Dict[str, dict]):
    """
    STEP 5 — Tampilkan ringkasan hasil semua stasiun.
    """
    logger.info("\n" + "=" * 60)
    logger.info("RINGKASAN HASIL")
    logger.info("=" * 60)

    rows = []
    for name, res in results.items():
        m = res.get("metrics", {})
        rows.append({
            "Stasiun": name,
            "MAE (°C)":  round(m.get("mae",  np.nan), 3),
            "RMSE (°C)": round(m.get("rmse", np.nan), 3),
            "MAPE (%)":  round(m.get("mape", np.nan), 2),
            "R²":        round(m.get("r2",   np.nan), 4),
            "CSV":       os.path.basename(res.get("csv_path", "")),
            "Plot":      os.path.basename(res.get("img_main", "")),
        })

    if rows:
        df_summary = pd.DataFrame(rows)
        print("\n" + df_summary.to_string(index=False))

        # Simpan ringkasan
        summary_path = os.path.join(DEFAULT_CONFIG["OUTPUT_DIR"], "summary_metrics.csv")
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"\n📋 Ringkasan disimpan → {summary_path}")
    else:
        logger.warning("Tidak ada hasil yang berhasil.")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(config: Optional[dict] = None) -> Dict[str, dict]:
    """
    Jalankan pipeline penuh.

    Args:
        config : dict konfigurasi (override DEFAULT_CONFIG). Kunci yang tersedia:
                 END_TRAIN, TARGET_DATE, STATIONS, HISTORICAL_START,
                 FORCE_REFRESH, SAVE_MODEL, SHOW_PLOTS, OUTPUT_DIR, LGBM_PARAMS.

    Returns:
        dict hasil per stasiun.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  BMKG Temperature Forecast — LightGBM Pipeline      ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"END_TRAIN   : {cfg['END_TRAIN']}")
    logger.info(f"TARGET_DATE : {cfg['TARGET_DATE']}")
    logger.info(f"STATIONS    : {cfg['STATIONS']}")
    logger.info(f"HISTORICAL  : mulai {cfg['HISTORICAL_START']}")

    # ── Validasi ──────────────────────────────────────────────────────────
    validate_dates(cfg["END_TRAIN"], cfg["TARGET_DATE"])

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    raw_data = step_fetch_data(
        station_names    = cfg["STATIONS"],
        historical_start = cfg["HISTORICAL_START"],
        force_refresh    = cfg["FORCE_REFRESH"],
    )

    # ── Step 2: Preprocessing ─────────────────────────────────────────────
    processed_data = step_preprocess(raw_data, cfg["END_TRAIN"], cfg["TARGET_DATE"])

    # ── Step 3 & 4: Training, Prediksi, Plot ─────────────────────────────
    results = step_train_predict(
        processed_data = processed_data,
        end_train      = cfg["END_TRAIN"],
        target_date    = cfg["TARGET_DATE"],
        save_model     = cfg["SAVE_MODEL"],
        show_plots     = cfg["SHOW_PLOTS"],
        lgbm_params    = cfg.get("LGBM_PARAMS"),
        output_dir     = cfg["OUTPUT_DIR"],
    )

    # ── Step 5: Ringkasan ─────────────────────────────────────────────────
    step_summary(results)

    logger.info("\n✅ Pipeline selesai.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Entry point via CLI."""
    parser = argparse.ArgumentParser(
        description="BMKG Temperature Forecast — LightGBM Pipeline"
    )
    parser.add_argument("--end-train",   default=DEFAULT_CONFIG["END_TRAIN"],
                        help="Tanggal akhir training (YYYY-MM-DD)")
    parser.add_argument("--target-date", default=DEFAULT_CONFIG["TARGET_DATE"],
                        help="Tanggal akhir prediksi (YYYY-MM-DD)")
    parser.add_argument("--stations",    nargs="+",
                        default=DEFAULT_CONFIG["STATIONS"],
                        help="Nama stasiun (spasi-separated)")
    parser.add_argument("--historical-start", default=DEFAULT_CONFIG["HISTORICAL_START"],
                        help="Mulai data historis (YYYY-MM-DD)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Paksa re-fetch data")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Jangan simpan model")
    parser.add_argument("--show-plots",  action="store_true",
                        help="Tampilkan grafik interaktif")
    args = parser.parse_args()

    config = {
        "END_TRAIN":        args.end_train,
        "TARGET_DATE":      args.target_date,
        "STATIONS":         args.stations,
        "HISTORICAL_START": args.historical_start,
        "FORCE_REFRESH":    args.force_refresh,
        "SAVE_MODEL":       not args.no_save_model,
        "SHOW_PLOTS":       args.show_plots,
    }
    run_pipeline(config)


if __name__ == "__main__":
    main()
