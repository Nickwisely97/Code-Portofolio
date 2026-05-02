"""
model.py
LightGBM model untuk prediksi suhu temperatur di Indonesia.

Parameter utama:
  END_TRAIN   : Tanggal akhir data training (str 'YYYY-MM-DD')
  TARGET_DATE : Tanggal akhir prediksi yang diinginkan (str 'YYYY-MM-DD')

Output:
  - Model terlatih
  - DataFrame prediksi (actual, predicted_actual, recorded_actual)
  - Grafik perbandingan
  - Metrics evaluasi (MAE, RMSE, MAPE, R²)
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
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, Dict, List
import joblib

from preprocessing import get_feature_columns, split_train_test, TARGET_COL

warnings.filterwarnings("ignore")
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
    LightGBM-based temperature forecasting model.

    Parameters
    ----------
    end_train   : str  — Tanggal akhir data training, format 'YYYY-MM-DD'
    target_date : str  — Tanggal akhir prediksi, format 'YYYY-MM-DD'
    station     : str  — Nama stasiun (untuk label grafik dan penyimpanan file)
    lgbm_params : dict — Hyperparameter LightGBM (opsional, override default)
    """

    def __init__(
        self,
        end_train:   str,
        target_date: str,
        station:     str = "Indonesia",
        lgbm_params: Optional[dict] = None,
    ):
        self.end_train   = end_train
        self.target_date = target_date
        self.station     = station
        self.params      = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}

        self.model        = None
        self.feature_cols: List[str] = []
        self.label_enc    = LabelEncoder()
        self.df_train_    = None
        self.df_test_     = None
        self.df_result_   = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Pisahkan X (features) dan y (target) dari DataFrame."""
        X = df[self.feature_cols].copy()
        y = df[TARGET_COL].copy()
        return X, y

    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "TemperatureForecastModel":
        """
        Latih model LightGBM.

        Args:
            df : DataFrame hasil preprocessing (output dari preprocessing.run_preprocessing).

        Returns:
            self
        """
        # Split train / test
        self.df_train_, self.df_test_ = split_train_test(
            df, self.end_train, self.target_date
        )

        if self.df_train_.empty:
            raise ValueError("Dataset training kosong. Periksa END_TRAIN dan data yang tersedia.")

        # Tentukan feature columns dari data training
        self.feature_cols = get_feature_columns(self.df_train_)
        logger.info(f"[Model] Feature columns: {len(self.feature_cols)} fitur")

        X_train, y_train = self._prepare_xy(self.df_train_)

        # Validasi dengan 10% terakhir dari training
        val_size   = max(1, int(len(X_train) * 0.10))
        X_val      = X_train.iloc[-val_size:]
        y_val      = y_train.iloc[-val_size:]
        X_tr       = X_train.iloc[:-val_size]
        y_tr       = y_train.iloc[:-val_size]

        logger.info(f"[Model] Training: {len(X_tr)} | Validation: {len(X_val)}")

        dtrain = lgb.Dataset(X_tr,  label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        # Pisahkan params training dari init params
        fit_params = {k: v for k, v in self.params.items()
                      if k not in ("n_estimators",)}

        self.model = lgb.train(
            params            = fit_params,
            train_set         = dtrain,
            num_boost_round   = self.params["n_estimators"],
            valid_sets        = [dtrain, dval],
            valid_names       = ["train", "valid"],
            callbacks         = callbacks,
        )

        # In-sample metrics
        y_pred_train = self.model.predict(X_train)
        self._log_metrics(y_train, y_pred_train, label="Train")

        return self

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Jalankan prediksi.

        Args:
            df : DataFrame test (opsional). Jika None, gunakan self.df_test_.

        Returns:
            DataFrame dengan kolom:
              - timestamp
              - actual            : nilai suhu aktual (dari data historis, bisa NaN untuk future)
              - predicted_actual  : prediksi model
              - recorded_actual   : nilai aktual yang ter-record (= actual jika ada)
              - residual          : selisih actual – predicted
        """
        if self.model is None:
            raise RuntimeError("Model belum dilatih. Jalankan .fit() terlebih dahulu.")

        if df is None:
            df = self.df_test_

        if df.empty:
            logger.warning("[Model] Test set kosong. Tidak ada prediksi.")
            return pd.DataFrame()

        X_test, y_test = self._prepare_xy(df)
        y_pred         = self.model.predict(X_test)

        result = pd.DataFrame({
            "timestamp":        df["timestamp"].values,
            "actual":           y_test.values,
            "predicted_actual": y_pred,
            "recorded_actual":  y_test.values,   # sama dengan actual (akan diisi NaN untuk future)
        })
        result["residual"] = result["actual"] - result["predicted_actual"]

        # Metrics test set
        mask = ~result["actual"].isna()
        if mask.sum() > 0:
            self._log_metrics(result.loc[mask, "actual"], result.loc[mask, "predicted_actual"], label="Test")

        self.df_result_ = result
        return result

    # ──────────────────────────────────────────────────────────────────────────
    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi untuk SELURUH rentang (train + test) sekaligus.
        Berguna untuk plot overlapping antara actual dan predicted.
        """
        if self.model is None:
            raise RuntimeError("Model belum dilatih.")

        X_all = df[self.feature_cols].copy()
        y_all = df[TARGET_COL].copy()
        y_pred = self.model.predict(X_all)

        result = pd.DataFrame({
            "timestamp":        df["timestamp"].values,
            "actual":           y_all.values,
            "predicted_actual": y_pred,
            "recorded_actual":  y_all.values,
            "is_train":         df["timestamp"] <= pd.to_datetime(self.end_train),
        })
        result["residual"] = result["actual"] - result["predicted_actual"]
        return result

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _log_metrics(y_true, y_pred, label: str = ""):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        # MAPE — hindari division by zero
        mask = np.abs(y_true) > 0.01
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
        logger.info(f"[Metrics-{label}] MAE={mae:.3f}°C | RMSE={rmse:.3f}°C | MAPE={mape:.2f}% | R²={r2:.4f}")
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    def get_metrics(self) -> Dict[str, float]:
        """Kembalikan metrics test set."""
        if self.df_result_ is None:
            raise RuntimeError("Jalankan .predict() terlebih dahulu.")
        mask = ~self.df_result_["actual"].isna()
        if mask.sum() == 0:
            return {}
        return self._log_metrics(
            self.df_result_.loc[mask, "actual"].values,
            self.df_result_.loc[mask, "predicted_actual"].values,
            label="Final",
        )

    # ──────────────────────────────────────────────────────────────────────────
    def plot_results(
        self,
        df_full:        Optional[pd.DataFrame] = None,
        last_n_days:    int = 90,
        save:           bool = True,
        show:           bool = False,
    ) -> str:
        """
        Plot tiga jalur:
          1. actual           — nilai suhu aktual (garis biru)
          2. predicted_actual — prediksi model (garis oranye putus-putus)
          3. recorded_actual  — marker titik untuk nilai yang ter-record

        Args:
            df_full      : Hasil predict_full() — seluruh periode. Jika None,
                           hanya menampilkan test window (self.df_result_).
            last_n_days  : Batasi tampilan N hari terakhir (0 = tampilkan semua).
            save         : Simpan gambar ke disk.
            show         : Tampilkan jendela interaktif (False untuk server/notebook).

        Returns:
            Path file gambar (str).
        """
        if df_full is None:
            df_plot = self.df_result_.copy() if self.df_result_ is not None else pd.DataFrame()
        else:
            df_plot = df_full.copy()

        if df_plot.empty:
            logger.warning("[Plot] Tidak ada data untuk diplot.")
            return ""

        # Filter N hari terakhir
        if last_n_days > 0:
            cutoff  = df_plot["timestamp"].max() - pd.Timedelta(days=last_n_days)
            df_plot = df_plot[df_plot["timestamp"] >= cutoff]

        # ── Layout ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            3, 1,
            figsize=(18, 13),
            gridspec_kw={"height_ratios": [3, 1, 1]},
            facecolor="#0f1117",
        )
        fig.suptitle(
            f"Prediksi Temperatur Indonesia — Stasiun: {self.station}\n"
            f"Train s/d {self.end_train}  |  Target: {self.target_date}",
            fontsize=14, color="white", fontweight="bold", y=1.01,
        )

        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#2a2d3a")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        # ── Panel 1: Perbandingan suhu ───────────────────────────────────────
        ax1 = axes[0]
        train_mask = df_plot.get("is_train", pd.Series(False, index=df_plot.index))

        # Actual (garis biru)
        ax1.plot(
            df_plot["timestamp"], df_plot["actual"],
            color="#4fc3f7", lw=1.2, alpha=0.85, label="Actual", zorder=2,
        )
        # Predicted (garis oranye putus-putus)
        ax1.plot(
            df_plot["timestamp"], df_plot["predicted_actual"],
            color="#ffb74d", lw=1.5, ls="--", alpha=0.90, label="Predicted", zorder=3,
        )
        # Recorded actual (titik hijau sparse)
        step = max(1, len(df_plot) // 300)
        ax1.scatter(
            df_plot["timestamp"].iloc[::step],
            df_plot["recorded_actual"].iloc[::step],
            color="#69f0ae", s=8, alpha=0.6, label="Recorded Actual", zorder=4,
        )
        # Shading train vs test
        if "is_train" in df_plot.columns:
            tr_end = df_plot.loc[df_plot["is_train"], "timestamp"].max()
            if pd.notna(tr_end):
                ax1.axvline(tr_end, color="#ef5350", lw=1.5, ls=":", alpha=0.8)
                ax1.text(tr_end, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 40,
                         "  Train End", color="#ef5350", fontsize=8, va="top")

        ax1.set_ylabel("Suhu (°C)", color="white")
        ax1.legend(loc="upper right", facecolor="#2a2d3a", labelcolor="white", fontsize=9)
        ax1.grid(True, alpha=0.15, color="gray")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # ── Panel 2: Residual ────────────────────────────────────────────────
        ax2 = axes[1]
        ax2.fill_between(
            df_plot["timestamp"], df_plot["residual"], 0,
            where=df_plot["residual"] >= 0,
            color="#69f0ae", alpha=0.5, label="Over-estimate",
        )
        ax2.fill_between(
            df_plot["timestamp"], df_plot["residual"], 0,
            where=df_plot["residual"] < 0,
            color="#ef5350", alpha=0.5, label="Under-estimate",
        )
        ax2.axhline(0, color="white", lw=0.8, alpha=0.5)
        ax2.set_ylabel("Residual (°C)", color="white")
        ax2.legend(loc="upper right", facecolor="#2a2d3a", labelcolor="white", fontsize=8)
        ax2.grid(True, alpha=0.1, color="gray")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # ── Panel 3: Feature Importance ──────────────────────────────────────
        ax3 = axes[2]
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importance(importance_type="gain"),
        }).nlargest(15, "importance")

        bars = ax3.barh(imp["feature"], imp["importance"], color="#7c4dff", alpha=0.8)
        ax3.set_xlabel("Gain", color="white")
        ax3.set_title("Top-15 Feature Importance (Gain)", color="white", fontsize=10)
        ax3.invert_yaxis()
        ax3.grid(True, axis="x", alpha=0.15, color="gray")

        plt.tight_layout()

        out_path = ""
        if save:
            fname    = f"{self.station.lower().replace(' ', '_')}_forecast.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"[Plot] Grafik disimpan ke: {out_path}")

        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    def plot_forecast_horizon(
        self,
        df_full: pd.DataFrame,
        horizon_days: int = 7,
        save: bool = True,
        show: bool = False,
    ) -> str:
        """
        Zoom-in grafik prediksi pada horizon beberapa hari ke depan.
        """
        cutoff   = pd.to_datetime(self.end_train)
        df_fore  = df_full[df_full["timestamp"] > cutoff].copy()
        df_last  = df_full[
            (df_full["timestamp"] > cutoff - pd.Timedelta(days=14)) &
            (df_full["timestamp"] <= cutoff)
        ].copy()

        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#2a2d3a")

        # Historical tail
        ax.plot(df_last["timestamp"], df_last["actual"],
                color="#4fc3f7", lw=1.5, label="Historical Actual")
        # Forecast
        ax.plot(df_fore["timestamp"].iloc[:horizon_days*24],
                df_fore["predicted_actual"].iloc[:horizon_days*24],
                color="#ffb74d", lw=2, ls="--", label=f"Forecast ({horizon_days}d)")
        ax.fill_between(
            df_fore["timestamp"].iloc[:horizon_days*24],
            df_fore["predicted_actual"].iloc[:horizon_days*24] - 1.5,
            df_fore["predicted_actual"].iloc[:horizon_days*24] + 1.5,
            color="#ffb74d", alpha=0.15, label="±1.5°C CI",
        )
        ax.axvline(cutoff, color="#ef5350", lw=1.5, ls=":", label="Train End")
        ax.set_title(
            f"Forecast {horizon_days}-Hari ke Depan — {self.station}",
            color="white", fontsize=13,
        )
        ax.set_ylabel("Suhu (°C)", color="white")
        ax.legend(facecolor="#2a2d3a", labelcolor="white")
        ax.grid(True, alpha=0.15, color="gray")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        out_path = ""
        if save:
            fname    = f"{self.station.lower().replace(' ', '_')}_horizon.png"
            out_path = os.path.join(OUTPUT_DIR, fname)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"[Plot] Horizon grafik disimpan ke: {out_path}")
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    def save_model(self, path: Optional[str] = None) -> str:
        """Simpan model ke disk."""
        if self.model is None:
            raise RuntimeError("Model belum dilatih.")
        if path is None:
            path = os.path.join(
                OUTPUT_DIR,
                f"lgbm_{self.station.lower().replace(' ', '_')}.pkl",
            )
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols,
                     "end_train": self.end_train, "target_date": self.target_date}, path)
        logger.info(f"[Model] Disimpan ke: {path}")
        return path

    @classmethod
    def load_model(cls, path: str) -> "TemperatureForecastModel":
        """Muat model dari disk."""
        data    = joblib.load(path)
        obj     = cls(end_train=data["end_train"], target_date=data["target_date"])
        obj.model        = data["model"]
        obj.feature_cols = data["feature_cols"]
        logger.info(f"[Model] Dimuat dari: {path}")
        return obj


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing import run_preprocessing

    # Data dummy
    np.random.seed(42)
    dates  = pd.date_range("2022-01-01", periods=24 * 365, freq="h")
    temps  = 27 + 3 * np.sin(2 * np.pi * dates.dayofyear / 365) + \
             2 * np.sin(2 * np.pi * dates.hour / 24) + np.random.randn(len(dates)) * 0.5
    df_raw = pd.DataFrame({"timestamp": dates, "temperature": temps,
                            "humidity": 80.0, "wind_speed": 10.0})

    df_proc = run_preprocessing(df_raw, station_name="Test")

    mdl = TemperatureForecastModel(
        end_train="2022-11-30",
        target_date="2022-12-31",
        station="Test",
    )
    mdl.fit(df_proc)
    result  = mdl.predict()
    df_full = mdl.predict_full(df_proc)
    mdl.plot_results(df_full, last_n_days=60)
    print("\nPrediksi selesai:")
    print(result[["timestamp", "actual", "predicted_actual", "residual"]].tail(5))
