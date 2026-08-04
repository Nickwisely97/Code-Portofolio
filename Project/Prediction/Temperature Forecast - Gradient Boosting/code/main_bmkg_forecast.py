"""
main_bmkg_forecast.py
Orchestrator utama pipeline prediksi temperatur Indonesia menggunakan data BMKG
dan Open-Meteo.

Pipeline (direct multi-horizon, forecast origin based):
  1. Tarik data (BMKG forecast + Open-Meteo historical)
  2. Preprocessing -> tabel origin-state per stasiun
  3. Bangun dataset: training origins (<= END_TRAIN), backtest origins
     (END_TRAIN..TARGET_DATE, rollback testing), dan satu live origin
     (default: data terbaru yang tersedia)
  4. Latih model LightGBM (satu model per stasiun, horizon_h sebagai fitur)
  5. Backtest walk-forward + forecast live 14 hari ke depan
  6. Plot hasil (skill vs horizon, forecast fan, feature importance) & simpan output
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict

# ── Import modul lokal ─────────────────────────────────────────────────────
from data_fetcher   import fetch_all_stations, STATIONS
from preprocessing  import (
    run_preprocessing, select_origins, build_climatology,
    build_direct_horizon_frame, get_feature_columns, TARGET_COL,
)
from model           import TemperatureForecastModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konfigurasi default ───────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Batas akhir origin training. None = otomatis: data terbaru dikurangi
    # TRAIN_BACKTEST_BUFFER_DAYS, supaya selalu relatif terhadap data yang
    # benar-benar tersedia dan tidak pernah basi seperti tanggal hardcoded.
    "END_TRAIN":          None,

    # Batas akhir jendela backtest (rollback testing). None = data terbaru
    # yang tersedia.
    "TARGET_DATE":        None,

    # Stasiun yang dijalankan.
    "STATIONS":           ["Jakarta", "Surabaya", "Bandung"],

    # Mulai tarik data historis dari tanggal ini.
    "HISTORICAL_START":   "2019-01-01",

    # Paksa re-fetch data meski cache sudah ada.
    "FORCE_REFRESH":      False,

    # Simpan model setelah training.
    "SAVE_MODEL":         True,

    # Tampilkan grafik interaktif (set True jika ada display).
    "SHOW_PLOTS":         False,

    # Output directory.
    "OUTPUT_DIR":         "output",

    # Jarak antar forecast origin saat membangun training/backtest set (jam).
    "ORIGIN_STRIDE_HOURS": 24,

    # Berapa jam ke depan diprediksi dari tiap origin (336 = 14 hari).
    "FORECAST_HORIZON_HOURS": 336,

    # Jarak default END_TRAIN dari TARGET_DATE saat END_TRAIN=None, supaya
    # ada cukup ruang untuk origin backtest antara keduanya.
    "TRAIN_BACKTEST_BUFFER_DAYS": 45,

    # Origin untuk forecast live (produk 14-hari-ke-depan). None = data
    # terbaru yang tersedia per stasiun ("as of sekarang").
    "LIVE_ORIGIN":        None,

    # Hyperparameter LightGBM (override jika diperlukan).
    "LGBM_PARAMS":        None,
}


# ─────────────────────────────────────────────────────────────────────────────
def resolve_dates(raw_data: Dict[str, pd.DataFrame], end_train, target_date, buffer_days: int) -> tuple:
    """
    Tentukan END_TRAIN/TARGET_DATE final. Kalau salah satu None, turunkan
    dari data terbaru yang benar-benar tersedia (bukan tanggal hardcoded)
    supaya pipeline tidak pernah menjalankan backtest yang sudah basi.
    """
    latest = max(
        (df["timestamp"].max() for df in raw_data.values() if not df.empty),
        default=None,
    )
    if latest is None:
        raise ValueError("No data available from any station — cannot resolve dates.")

    td = pd.to_datetime(target_date) if target_date else latest
    et = pd.to_datetime(end_train) if end_train else (td - pd.Timedelta(days=buffer_days))

    return et.strftime("%Y-%m-%d"), td.strftime("%Y-%m-%d")


def validate_dates(end_train: str, target_date: str, horizon_hours: int):
    """Pastikan ada cukup jarak antara END_TRAIN dan TARGET_DATE untuk backtest penuh."""
    et = pd.to_datetime(end_train)
    td = pd.to_datetime(target_date)
    min_gap = pd.Timedelta(hours=horizon_hours)
    if td < et + min_gap:
        raise ValueError(
            f"TARGET_DATE ({target_date}) must be at least {horizon_hours}h "
            f"({horizon_hours // 24}d) after END_TRAIN ({end_train}) so there's room "
            f"for at least one full-horizon backtest origin."
        )
    logger.info(
        f"Period: train origins <= {end_train} | backtest origins {end_train} -> "
        f"{target_date} (horizon {horizon_hours}h)"
    )


# ─────────────────────────────────────────────────────────────────────────────
def step_fetch_data(
    station_names:     list,
    historical_start:  str,
    force_refresh:     bool,
) -> Dict[str, pd.DataFrame]:
    """STEP 1 — Tarik data dari BMKG & Open-Meteo. Mengembalikan dict {station: DataFrame raw}."""
    logger.info("=" * 60)
    logger.info("STEP 1: Fetch Data")
    logger.info("=" * 60)

    target_stations = {k: v for k, v in STATIONS.items() if k in station_names}
    if not target_stations:
        raise ValueError(f"No valid stations. Options: {list(STATIONS.keys())}")

    data = fetch_all_stations(
        stations         = target_stations,
        historical_start = historical_start,
        force_refresh    = force_refresh,
    )
    for name, df in data.items():
        if df.empty:
            logger.warning(f"  !  Data for {name} is empty!")
        else:
            logger.info(
                f"  ok  {name}: {len(df):,} rows | "
                f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()}"
            )
    return data


# ─────────────────────────────────────────────────────────────────────────────
def step_preprocess(
    raw_data:     Dict[str, pd.DataFrame],
    end_train:    str,
    stride_hours: int,
) -> Dict[str, pd.DataFrame]:
    """STEP 2 — Preprocessing (origin-state) per stasiun."""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing")
    logger.info("=" * 60)

    processed = {}
    for name, df_raw in raw_data.items():
        if df_raw.empty:
            logger.warning(f"  !  {name}: empty data, skipped.")
            continue
        try:
            df_proc = run_preprocessing(df_raw, station_name=name, drop_na=True)
            n_train_origins = len(select_origins(df_proc, end=end_train, stride_hours=stride_hours))
            if n_train_origins < 30:
                logger.warning(f"  !  {name}: only {n_train_origins} training origins — may not be enough.")
            processed[name] = df_proc
            logger.info(
                f"  ok  {name}: {len(df_proc):,} origin rows | "
                f"{len(get_feature_columns(df_proc))} state features | "
                f"{n_train_origins} training origins"
            )
        except Exception as e:
            logger.error(f"  x  {name}: preprocessing failed — {e}")

    return processed


# ─────────────────────────────────────────────────────────────────────────────
def step_build_datasets(
    processed_data: Dict[str, pd.DataFrame],
    end_train:      str,
    target_date:    str,
    stride_hours:   int,
    horizon_hours:  int,
    live_origin:    Optional[str] = None,
) -> Dict[str, dict]:
    """
    STEP 3 — Bangun training / backtest / live direct-horizon frames per
    stasiun (rolling-origin expansion, lihat preprocessing.build_direct_horizon_frame).
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Build Direct-Horizon Datasets (Rolling-Origin)")
    logger.info("=" * 60)

    backtest_end = pd.to_datetime(target_date) - pd.Timedelta(hours=horizon_hours)
    datasets = {}

    for name, df_state in processed_data.items():
        try:
            df_actual = df_state[["timestamp", TARGET_COL]].copy()
            clim = build_climatology(df_actual, as_of=end_train)

            train_origins    = select_origins(df_state, end=end_train, stride_hours=stride_hours)
            backtest_origins = select_origins(df_state, start=end_train, end=backtest_end, stride_hours=stride_hours)

            df_train    = build_direct_horizon_frame(df_state, df_actual, train_origins, clim, horizon_hours)
            df_backtest = build_direct_horizon_frame(df_state, df_actual, backtest_origins, clim, horizon_hours)

            live_cutoff = pd.to_datetime(live_origin) if live_origin else df_state["timestamp"].max()
            live_candidates = select_origins(df_state, end=live_cutoff, stride_hours=1)
            if live_candidates.empty:
                logger.warning(f"  !  {name}: no valid live origin at/before {live_cutoff} — skipping live forecast.")
                df_live = pd.DataFrame()
            else:
                df_live = build_direct_horizon_frame(
                    df_state, df_actual, live_candidates.iloc[[-1]], clim, horizon_hours,
                )

            datasets[name] = {
                "train": df_train, "backtest": df_backtest, "live": df_live,
                "climatology": clim, "actual": df_actual,
            }
            logger.info(
                f"  ok  {name}: train={len(df_train):,} rows | "
                f"backtest={len(df_backtest):,} rows | live={len(df_live):,} rows"
            )
        except Exception as e:
            logger.error(f"  x  {name}: dataset build failed — {e}", exc_info=True)

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
def step_train_predict(
    datasets:    Dict[str, dict],
    end_train:   str,
    target_date: str,
    save_model:  bool,
    show_plots:  bool,
    lgbm_params: Optional[dict],
    output_dir:  str,
) -> Dict[str, dict]:
    """STEP 4 — Training, backtest, live forecast, dan plotting per stasiun."""
    logger.info("=" * 60)
    logger.info("STEP 4: Train, Backtest, Live Forecast, Plot")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, ds in datasets.items():
        logger.info(f"\n-- Station: {name} --------------------------------")
        try:
            mdl = TemperatureForecastModel(
                station     = name,
                end_train   = end_train,
                target_date = target_date,
                lgbm_params = lgbm_params,
            )
            mdl.climatology_ = ds["climatology"]

            logger.info("  >  Training LightGBM model ...")
            mdl.fit(ds["train"])

            backtest_result = mdl.predict_backtest(ds["backtest"])
            metrics    = mdl.get_metrics() if not backtest_result.empty else {}
            by_horizon = mdl.metrics_by_horizon(backtest_result)

            if metrics:
                logger.info(
                    f"  ..  MAE={metrics.get('mae', float('nan')):.3f}°C | "
                    f"RMSE={metrics.get('rmse', float('nan')):.3f}°C | "
                    f"MAPE={metrics.get('mape', float('nan')):.2f}% | "
                    f"R²={metrics.get('r2', float('nan')):.4f}"
                )
            else:
                logger.warning("  !  No backtest metrics available (empty backtest window).")

            backtest_csv = os.path.join(output_dir, f"{name.lower()}_backtest_predictions.csv")
            backtest_result.to_csv(backtest_csv, index=False)

            live_result = pd.DataFrame()
            live_csv    = ""
            if not ds["live"].empty:
                live_result = mdl.predict_live(ds["live"])
                live_csv    = os.path.join(output_dir, f"{name.lower()}_forecast_next14d.csv")
                live_result.to_csv(live_csv, index=False)
                logger.info(f"  ..  Live 14-day forecast saved -> {live_csv}")

            # Plots
            img_skill = mdl.plot_skill_by_horizon(backtest_result, show=show_plots)
            img_fan   = ""
            if not live_result.empty:
                img_fan = mdl.plot_forecast_fan(ds["actual"], live_result, show=show_plots)
            img_importance = mdl.plot_feature_importance(show=show_plots)

            model_path = mdl.save_model() if save_model else ""

            results[name] = {
                "model":          mdl,
                "metrics":        metrics,
                "by_horizon":     by_horizon,
                "backtest_csv":   backtest_csv,
                "live_csv":       live_csv,
                "img_skill":      img_skill,
                "img_fan":        img_fan,
                "img_importance": img_importance,
                "model_path":     model_path,
            }

        except Exception as e:
            logger.error(f"  x  {name}: failed — {e}", exc_info=True)

    return results


# ─────────────────────────────────────────────────────────────────────────────
def step_summary(results: Dict[str, dict], output_dir: str):
    """STEP 5 — Ringkasan hasil semua stasiun."""
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    rows = []
    for name, res in results.items():
        m = res.get("metrics", {})
        rows.append({
            "Station":    name,
            "MAE (°C)":   round(m.get("mae",  np.nan), 3),
            "RMSE (°C)":  round(m.get("rmse", np.nan), 3),
            "MAPE (%)":   round(m.get("mape", np.nan), 2),
            "R²":         round(m.get("r2",   np.nan), 4),
            "Live CSV":   os.path.basename(res.get("live_csv", "")),
            "Skill Plot": os.path.basename(res.get("img_skill", "")),
        })

    if not rows:
        logger.warning("No successful results.")
        return

    df_summary = pd.DataFrame(rows)
    print("\n" + df_summary.to_string(index=False))

    # Bug fix: write to the *configured* output_dir, not a hardcoded default —
    # previously this always used DEFAULT_CONFIG["OUTPUT_DIR"] regardless of
    # what the caller actually passed via config/CLI.
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved -> {summary_path}")

    for name, res in results.items():
        by_h = res.get("by_horizon")
        if by_h is not None and not by_h.empty:
            path = os.path.join(output_dir, f"{name.lower()}_metrics_by_horizon.csv")
            by_h.to_csv(path, index=False)
            logger.info(f"  ..  {name}: metrics-by-horizon-day saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(config: Optional[dict] = None) -> Dict[str, dict]:
    """
    Jalankan pipeline penuh.

    Args:
        config : dict konfigurasi (override DEFAULT_CONFIG). Kunci yang tersedia:
                 END_TRAIN, TARGET_DATE, STATIONS, HISTORICAL_START,
                 FORCE_REFRESH, SAVE_MODEL, SHOW_PLOTS, OUTPUT_DIR,
                 ORIGIN_STRIDE_HOURS, FORECAST_HORIZON_HOURS,
                 TRAIN_BACKTEST_BUFFER_DAYS, LIVE_ORIGIN, LGBM_PARAMS.

    Returns:
        dict hasil per stasiun.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info("=" * 60)
    logger.info("BMKG Temperature Forecast -- LightGBM Direct-Horizon Pipeline")
    logger.info("=" * 60)
    logger.info(f"STATIONS    : {cfg['STATIONS']}")
    logger.info(f"HISTORICAL  : mulai {cfg['HISTORICAL_START']}")

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    raw_data = step_fetch_data(
        station_names    = cfg["STATIONS"],
        historical_start = cfg["HISTORICAL_START"],
        force_refresh    = cfg["FORCE_REFRESH"],
    )

    # ── Resolve & validate dates (relative to data actually available) ────
    end_train, target_date = resolve_dates(
        raw_data, cfg["END_TRAIN"], cfg["TARGET_DATE"], cfg["TRAIN_BACKTEST_BUFFER_DAYS"],
    )
    validate_dates(end_train, target_date, cfg["FORECAST_HORIZON_HOURS"])
    logger.info(f"END_TRAIN   : {end_train}")
    logger.info(f"TARGET_DATE : {target_date}")

    # ── Step 2: Preprocessing ─────────────────────────────────────────────
    processed_data = step_preprocess(raw_data, end_train, cfg["ORIGIN_STRIDE_HOURS"])

    # ── Step 3: Build datasets ────────────────────────────────────────────
    datasets = step_build_datasets(
        processed_data, end_train, target_date,
        stride_hours  = cfg["ORIGIN_STRIDE_HOURS"],
        horizon_hours = cfg["FORECAST_HORIZON_HOURS"],
        live_origin   = cfg["LIVE_ORIGIN"],
    )

    # ── Step 4: Train, Backtest, Live Forecast, Plot ─────────────────────
    results = step_train_predict(
        datasets    = datasets,
        end_train   = end_train,
        target_date = target_date,
        save_model  = cfg["SAVE_MODEL"],
        show_plots  = cfg["SHOW_PLOTS"],
        lgbm_params = cfg.get("LGBM_PARAMS"),
        output_dir  = cfg["OUTPUT_DIR"],
    )

    # ── Step 5: Ringkasan ─────────────────────────────────────────────────
    step_summary(results, cfg["OUTPUT_DIR"])

    logger.info("\nPipeline selesai.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Entry point via CLI."""
    parser = argparse.ArgumentParser(
        description="BMKG Temperature Forecast — LightGBM Direct-Horizon Pipeline"
    )
    parser.add_argument("--end-train",   default=None,
                        help="Batas akhir origin training (YYYY-MM-DD). Default: otomatis dari data terbaru.")
    parser.add_argument("--target-date", default=None,
                        help="Batas akhir jendela backtest (YYYY-MM-DD). Default: data terbaru yang tersedia.")
    parser.add_argument("--stations",    nargs="+",
                        default=DEFAULT_CONFIG["STATIONS"],
                        help="Nama stasiun (spasi-separated)")
    parser.add_argument("--historical-start", default=DEFAULT_CONFIG["HISTORICAL_START"],
                        help="Mulai data historis (YYYY-MM-DD)")
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_CONFIG["FORECAST_HORIZON_HOURS"],
                        help="Berapa jam ke depan diprediksi per origin (default 336 = 14 hari)")
    parser.add_argument("--origin-stride-hours", type=int, default=DEFAULT_CONFIG["ORIGIN_STRIDE_HOURS"],
                        help="Jarak antar forecast origin saat membangun dataset (default 24 = harian)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Paksa re-fetch data")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Jangan simpan model")
    parser.add_argument("--show-plots",  action="store_true",
                        help="Tampilkan grafik interaktif")
    args = parser.parse_args()

    config = {
        "END_TRAIN":              args.end_train,
        "TARGET_DATE":            args.target_date,
        "STATIONS":               args.stations,
        "HISTORICAL_START":       args.historical_start,
        "FORECAST_HORIZON_HOURS": args.horizon_hours,
        "ORIGIN_STRIDE_HOURS":    args.origin_stride_hours,
        "FORCE_REFRESH":          args.force_refresh,
        "SAVE_MODEL":             not args.no_save_model,
        "SHOW_PLOTS":             args.show_plots,
    }
    run_pipeline(config)


if __name__ == "__main__":
    main()
