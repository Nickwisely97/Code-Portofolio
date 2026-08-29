# Temperature Forecast — LightGBM Multi-Horizon

**CV skill represented:** Forecasting & Time-Series (LightGBM, feature engineering for time-series & categorical data).

## Problem
Hourly, 14-day-ahead temperature forecasts for Indonesian cities, combining live BMKG forecasts with Open-Meteo (ERA5) historical data.

## Method
- **Direct multi-horizon**: one model predicts every hour from T0+1h to T0+336h at once (`horizon_h` as a feature), not an autoregressive per-hour model.
- Features split into origin-state (known at/before T0: lags, rolling stats, momentum) and target-calendar (always knowable in advance: seasonality, climatology) — deliberately excludes same-timestamp weather, which isn't knowable days ahead.
- Validated with walk-forward rollback testing: many historical origins, each scored against what actually happened, broken down by horizon day to confirm error grows honestly with distance rather than leaking.

## Result
Backtest (Jakarta): MAE ≈ 0.88°C, RMSE ≈ 1.16°C, R² ≈ 0.78.

## Structure
`code/` is a plain importable Python package (`data_fetcher.py`, `preprocessing.py`, `model.py`, `main_bmkg_forecast.py`) — no notebooks inside it. `temperature_forecast_pipeline.ipynb`, at the project root, is the orchestrator notebook that walks through the pipeline step by step using those modules; it's kept outside `code/` deliberately since `code/` holds only library modules.

## How to run
- Interactive: open `temperature_forecast_pipeline.ipynb` at the project root and run top to bottom.
- CLI: `python code/main_bmkg_forecast.py --stations Jakarta`
- Data is cached to `data/*.parquet` (re-fetches from BMKG/Open-Meteo only if missing or `--force-refresh`). Outputs (metrics, forecasts, plots, saved model) go to `result/`.
