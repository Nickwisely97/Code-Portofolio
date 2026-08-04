"""
model.py
LightGBM model untuk prediksi suhu temperatur di Indonesia.

Model ini adalah "direct multi-horizon": satu model memprediksi suhu pada
jam manapun dari 1 sampai `horizon_hours` ke depan dari sebuah forecast
origin T0 sekaligus, dengan `horizon_h` sebagai salah satu fitur. Data
latih/evaluasi/live-forecast dibangun di preprocessing.py
(build_direct_horizon_frame) — kelas di sini murni urusan training,
prediksi, metrik, dan plotting terhadap frame yang sudah jadi.

Output:
  - Model terlatih
  - Metrik backtest (walk-forward, per hari horizon)
  - Forecast live (14 hari ke depan dari origin terbaru)
  - Grafik: skill vs horizon, forecast fan, feature importance
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, Tuple, Dict, List
import joblib

from preprocessing import get_feature_columns, TARGET_COL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Default Hyperparameters LightGBM ───────────────────────────────────────
DEFAULT_LGBM_PARAMS = {
    "objective":        "regression",
    "metric":           ["mae", "rmse"],
    "boosting_type":    "gbdt",
    "num_leaves":       127,
    "max_depth":        -1,
    "learning_rate":    0.05,
    "n_estimators":     1500,
    "min_child_samples": 20,
    "subsample":        0.85,
    "subsample_freq":   1,
    "colsample_bytree": 0.85,
    "reg_alpha":        0.1,
    "reg_lambda":       0.2,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

OUTPUT_DIR = "output"


# ─── Kelas Model ─────────────────────────────────────────────────────────────
class TemperatureForecastModel:
    """
    LightGBM direct multi-horizon temperature forecasting model.

    Parameters
    ----------
    station     : str  — Nama stasiun (untuk label grafik dan penyimpanan file)
    end_train   : str  — Metadata label saja (batas origin training), untuk judul grafik
    target_date : str  — Metadata label saja (batas jendela backtest), untuk judul grafik
    lgbm_params : dict — Hyperparameter LightGBM (opsional, override default)
    """

    def __init__(
        self,
        station:     str = "Indonesia",
        end_train:   Optional[str] = None,
        target_date: Optional[str] = None,
        lgbm_params: Optional[dict] = None,
    ):
        self.station     = station
        self.end_train   = end_train
        self.target_date = target_date
        self.params      = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}

        self.model         = None
        self.feature_cols: List[str] = []
        self.climatology_   = None
        self.df_backtest_   = None
        self.df_live_       = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Pisahkan X (features) dan y (target) dari DataFrame."""
        X = df[self.feature_cols].copy()
        y = df[TARGET_COL].copy()
        return X, y

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> "TemperatureForecastModel":
        """
        Latih model LightGBM pada direct-horizon training frame (hasil
        preprocessing.build_direct_horizon_frame, semua barisnya punya
        actual/y karena origin-nya cukup lama).

        Validasi dipisah per ORIGIN (bukan per baris): 10% origin paling
        akhir ditahan sebagai validation set, supaya baris-baris satu
        origin yang sama tidak bocor antara train dan validation.
        """
        if df_train.empty:
            raise ValueError("Training dataset is empty. Check origin selection / date range.")

        self.feature_cols = get_feature_columns(df_train)
        logger.info(f"[Model] Feature columns: {len(self.feature_cols)} features")

        origins    = np.sort(df_train["origin"].unique())
        n_val      = max(1, int(len(origins) * 0.10))
        val_origin_set = set(origins[-n_val:])
        is_val     = df_train["origin"].isin(val_origin_set)

        df_tr, df_val = df_train[~is_val], df_train[is_val]
        X_tr, y_tr    = self._prepare_xy(df_tr)
        X_val, y_val  = self._prepare_xy(df_val)

        logger.info(
            f"[Model] Origins: {len(origins)} total, {len(origins) - n_val} train / {n_val} val | "
            f"Rows: {len(X_tr)} train / {len(X_val)} val"
        )

        dtrain = lgb.Dataset(X_tr,  label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        fit_params = {k: v for k, v in self.params.items() if k not in ("n_estimators",)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                params          = fit_params,
                train_set       = dtrain,
                num_boost_round = self.params["n_estimators"],
                valid_sets      = [dtrain, dval],
                valid_names     = ["train", "valid"],
                callbacks       = callbacks,
            )

        self._log_metrics(y_tr, self.model.predict(X_tr), label="Train")
        self._log_metrics(y_val, self.model.predict(X_val), label="Validation")

        return self

    # ──────────────────────────────────────────────────────────────────────────
    def _predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prediksi mentah untuk sebuah direct-horizon frame, apa adanya."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Run .fit() first.")
        if df.empty:
            return pd.DataFrame(columns=["origin", "horizon_h", "timestamp", "predicted_actual"])

        X = df[self.feature_cols].copy()
        y_pred = self.model.predict(X)

        result = df[["origin", "horizon_h", "timestamp"]].copy()
        result["predicted_actual"] = y_pred
        if TARGET_COL in df.columns:
            result["actual"]   = df[TARGET_COL].values
            result["residual"] = result["actual"] - result["predicted_actual"]
        return result.reset_index(drop=True)

    def predict_backtest(self, df_backtest: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi + evaluasi untuk sebuah backtest frame (rollback testing):
        banyak origin historis, masing-masing dengan target yang sudah
        benar-benar terjadi, jadi bisa dibandingkan dengan actual.
        """
        result = self._predict_frame(df_backtest)
        if result.empty:
            logger.warning("[Model] Backtest frame is empty. No predictions made.")
            self.df_backtest_ = result
            return result

        mask = ~result["actual"].isna()
        if mask.sum() > 0:
            self._log_metrics(result.loc[mask, "actual"], result.loc[mask, "predicted_actual"], label="Backtest")
        self.df_backtest_ = result
        return result

    def predict_live(self, df_live: pd.DataFrame) -> pd.DataFrame:
        """
        Forecast produk sebenarnya: SATU origin ("as of" hari ini/terbaru),
        diperluas ke seluruh horizon (biasanya 14 hari) — target-nya adalah
        masa depan sungguhan, jadi `actual` akan NaN.
        """
        result = self._predict_frame(df_live)
        self.df_live_ = result
        return result

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _log_metrics(y_true, y_pred, label: str = "") -> Dict[str, float]:
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        mask = np.abs(y_true_arr) > 0.01
        mape = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100 if mask.sum() > 0 else np.nan
        logger.info(f"[Metrics-{label}] MAE={mae:.3f}°C | RMSE={rmse:.3f}°C | MAPE={mape:.2f}% | R²={r2:.4f}")
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    def get_metrics(self) -> Dict[str, float]:
        """Kembalikan metrics keseluruhan dari backtest terakhir."""
        if self.df_backtest_ is None:
            raise RuntimeError("Run .predict_backtest() first.")
        mask = ~self.df_backtest_["actual"].isna()
        if mask.sum() == 0:
            return {}
        return self._log_metrics(
            self.df_backtest_.loc[mask, "actual"].values,
            self.df_backtest_.loc[mask, "predicted_actual"].values,
            label="Backtest-Final",
        )

    def metrics_by_horizon(self, df_backtest: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Metrik backtest per HARI horizon (1..14) — diagnostik utama untuk
        memastikan model benar-benar berperilaku seperti forecast jarak
        jauh (error naik seiring horizon), bukan persistence yang menyamar.
        """
        df = df_backtest if df_backtest is not None else self.df_backtest_
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.dropna(subset=["actual"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["horizon_day"] = ((df["horizon_h"] - 1) // 24 + 1).astype(int)

        rows = []
        for day, g in df.groupby("horizon_day"):
            rows.append({
                "horizon_day": int(day),
                "n":           len(g),
                "mae":         mean_absolute_error(g["actual"], g["predicted_actual"]),
                "rmse":        float(np.sqrt(mean_squared_error(g["actual"], g["predicted_actual"]))),
                "r2":          r2_score(g["actual"], g["predicted_actual"]) if len(g) > 1 else np.nan,
            })
        return pd.DataFrame(rows).sort_values("horizon_day").reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _style_dark_axes(ax):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2a2d3a")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    def plot_skill_by_horizon(
        self,
        df_backtest: Optional[pd.DataFrame] = None,
        save: bool = True,
        show: bool = False,
    ) -> str:
        """
        Plot utama untuk memvalidasi bahwa ini forecast jarak jauh yang
        jujur: MAE & RMSE per hari horizon (1..14). Error yang naik
        bertahap = wajar; error yang flat/nyaris nol di semua hari = tanda
        ada leakage.
        """
        by_horizon = self.metrics_by_horizon(df_backtest)
        if by_horizon.empty:
            logger.warning("[Plot] No backtest data to plot skill-by-horizon.")
            return ""

        fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0f1117")
        self._style_dark_axes(ax)

        ax.plot(by_horizon["horizon_day"], by_horizon["mae"],
                color="#4fc3f7", marker="o", lw=2, label="MAE (°C)")
        ax.plot(by_horizon["horizon_day"], by_horizon["rmse"],
                color="#ffb74d", marker="o", lw=2, label="RMSE (°C)")
        ax.set_title(f"Backtest Skill vs Horizon Day — {self.station}", color="white", fontsize=13)
        ax.set_xlabel("Horizon (days ahead)", color="white")
        ax.set_ylabel("Error (°C)", color="white")
        ax.set_xticks(by_horizon["horizon_day"])
        ax.legend(facecolor="#2a2d3a", labelcolor="white")
        ax.grid(True, alpha=0.15, color="gray")

        plt.tight_layout()
        out_path = ""
        if save:
            fname = f"{self.station.lower().replace(' ', '_')}_skill_by_horizon.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"[Plot] Skill-by-horizon chart saved to: {out_path}")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    def plot_forecast_fan(
        self,
        df_history: pd.DataFrame,
        df_forecast: pd.DataFrame,
        history_days: int = 21,
        save: bool = True,
        show: bool = False,
    ) -> str:
        """
        Plot satu origin: histori aktual sebelum origin + kipas forecast
        14 hari ke depan. Kalau df_forecast punya kolom 'actual' terisi
        (origin backtest), aktual sungguhan ikut di-overlay untuk perbandingan.

        Args:
            df_history  : DataFrame timestamp/temperature aktual sebelum origin.
            df_forecast : Hasil predict_live()/predict_backtest() untuk SATU origin.
        """
        if df_forecast.empty:
            logger.warning("[Plot] No forecast data to plot.")
            return ""

        origin = df_forecast["origin"].iloc[0]
        cutoff = pd.to_datetime(origin) - pd.Timedelta(days=history_days)
        hist   = df_history[(df_history["timestamp"] > cutoff) & (df_history["timestamp"] <= origin)]

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
        self._style_dark_axes(ax)

        ax.plot(hist["timestamp"], hist[TARGET_COL],
                color="#4fc3f7", lw=1.5, label="Historical Actual")
        ax.plot(df_forecast["timestamp"], df_forecast["predicted_actual"],
                color="#ffb74d", lw=2, ls="--", label="Forecast (14d)")
        ax.fill_between(
            df_forecast["timestamp"],
            df_forecast["predicted_actual"] - 1.5,
            df_forecast["predicted_actual"] + 1.5,
            color="#ffb74d", alpha=0.15, label="±1.5°C band",
        )
        if "actual" in df_forecast.columns and df_forecast["actual"].notna().any():
            ax.scatter(df_forecast["timestamp"], df_forecast["actual"],
                       color="#69f0ae", s=10, alpha=0.7, label="Actual outcome (backtest)")

        ax.axvline(pd.to_datetime(origin), color="#ef5350", lw=1.5, ls=":", label="Forecast Origin")
        ax.set_title(f"14-Day Forecast from {pd.to_datetime(origin).date()} — {self.station}",
                     color="white", fontsize=13)
        ax.set_ylabel("Temperature (°C)", color="white")
        ax.legend(facecolor="#2a2d3a", labelcolor="white")
        ax.grid(True, alpha=0.15, color="gray")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        out_path = ""
        if save:
            fname = f"{self.station.lower().replace(' ', '_')}_forecast_fan.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"[Plot] Forecast-fan chart saved to: {out_path}")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    def plot_feature_importance(
        self,
        top_n: int = 15,
        save: bool = True,
        show: bool = False,
    ) -> str:
        """Top-N feature importance (gain) dari model terlatih."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).nlargest(top_n, "importance")

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1117")
        self._style_dark_axes(ax)
        ax.barh(imp["feature"], imp["importance"], color="#7c4dff", alpha=0.85)
        ax.set_title(f"Top-{top_n} Feature Importance (Gain) — {self.station}", color="white", fontsize=12)
        ax.set_xlabel("Gain", color="white")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.15, color="gray")

        plt.tight_layout()
        out_path = ""
        if save:
            fname = f"{self.station.lower().replace(' ', '_')}_feature_importance.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"[Plot] Feature importance chart saved to: {out_path}")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    def save_model(self, path: Optional[str] = None) -> str:
        """Simpan model + metadata (feature cols, climatology table) ke disk."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        if path is None:
            path = os.path.join(OUTPUT_DIR, f"lgbm_{self.station.lower().replace(' ', '_')}.pkl")
        joblib.dump({
            "model":        self.model,
            "feature_cols": self.feature_cols,
            "climatology":  self.climatology_,
            "station":      self.station,
            "end_train":    self.end_train,
            "target_date":  self.target_date,
        }, path)
        logger.info(f"[Model] Saved to: {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> "TemperatureForecastModel":
        """Muat model dari disk."""
        data = joblib.load(path)
        obj  = cls(
            station     = data.get("station", "Indonesia"),
            end_train   = data.get("end_train"),
            target_date = data.get("target_date"),
        )
        obj.model         = data["model"]
        obj.feature_cols  = data["feature_cols"]
        obj.climatology_  = data.get("climatology")
        logger.info(f"[Model] Loaded from: {path}")
        return obj


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import (
        run_preprocessing, select_origins, build_climatology,
        build_direct_horizon_frame, FORECAST_HORIZON_HOURS,
    )

    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=24 * 500, freq="h")
    temps = (27 + 3 * np.sin(2 * np.pi * dates.dayofyear / 365)
                + 2 * np.sin(2 * np.pi * dates.hour / 24)
                + np.random.randn(len(dates)) * 0.5)
    df_raw = pd.DataFrame({
        "timestamp": dates, "temperature": temps,
        "humidity": 80.0, "wind_speed": 10.0, "precipitation": 0.0,
    })

    df_state = run_preprocessing(df_raw.copy(), station_name="Test")

    end_train   = dates.max() - pd.Timedelta(days=45)
    target_date = dates.max()

    train_origins    = select_origins(df_state, end=end_train, stride_hours=24)
    backtest_origins = select_origins(df_state, start=end_train, end=target_date - pd.Timedelta(hours=FORECAST_HORIZON_HOURS), stride_hours=24)

    clim = build_climatology(df_raw, as_of=end_train)

    df_train    = build_direct_horizon_frame(df_state, df_raw, train_origins, clim)
    df_backtest = build_direct_horizon_frame(df_state, df_raw, backtest_origins, clim)

    mdl = TemperatureForecastModel(station="Test", end_train=str(end_train.date()), target_date=str(target_date.date()))
    mdl.climatology_ = clim
    mdl.fit(df_train)

    result = mdl.predict_backtest(df_backtest)
    print("\nBacktest skill by horizon day:")
    print(mdl.metrics_by_horizon().to_string(index=False))

    # Live forecast from the very latest origin available.
    live_origin = select_origins(df_state, stride_hours=1).iloc[[-1]]
    df_live = build_direct_horizon_frame(df_state, df_raw, live_origin, clim)
    live_result = mdl.predict_live(df_live)
    print(f"\nLive forecast rows: {len(live_result)} (actual should be all NaN — real future)")
    print(live_result.head(3).to_string())

    mdl.plot_skill_by_horizon(result)
    mdl.plot_forecast_fan(df_raw, live_result)
    mdl.plot_feature_importance()
    print("\nSmoke test complete.")
