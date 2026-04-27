import os
import logging
import joblib
import glob

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime

from app.core.config import settings
from app.services.data_service import data_service
from app.services.feature_service import feature_service, FEATURE_COLS

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# FIX #3: File-based lock to prevent multiple gunicorn workers from training simultaneously
try:
    import filelock
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    logger.warning("filelock not installed. Multi-worker race condition protection disabled. Run: pip install filelock")


class ModelService:
    def __init__(self):
        self._model_cache = {}
        self._training_in_progress = {}

    def _model_path(self, league_code: str, timestamp: bool = True) -> str:
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(MODELS_DIR, f"{league_code}_xgb_{ts}.joblib")
        return os.path.join(MODELS_DIR, f"{league_code}_xgb_*.joblib")

    def _load_model(self, league_code: str):
        payload = None
        if league_code in self._model_cache:
            payload = self._model_cache[league_code]
        else:
            search_pattern = self._model_path(league_code, timestamp=False)
            files = glob.glob(search_pattern)

            if files:
                # FIX #8: Sort by filename (timestamp in name) instead of ctime
                # ctime can be identical on Render deploy making selection non-deterministic
                latest_file = max(files, key=lambda f: os.path.basename(f))
                try:
                    loaded = joblib.load(latest_file)
                    if "xg_home" in loaded:
                        payload = loaded
                        self._model_cache[league_code] = payload
                        logger.info(f"Loaded model for {league_code}: {os.path.basename(latest_file)}")
                except Exception as e:
                    logger.error(f"Error loading model for {league_code} from {latest_file}: {e}")

        if payload:
            if "trained_at" in payload:
                trained_at = datetime.fromisoformat(payload["trained_at"])
                age_days = (datetime.now() - trained_at).total_seconds() / 86400.0
                payload["model_age_days"] = round(age_days, 2)
                payload["is_stale"] = age_days > 7.0
            else:
                payload["model_age_days"] = 999.0
                payload["is_stale"] = True

        return payload

    def _train_and_save(self, league_code: str, matches: list):
        """Train XGBRegressors for expected goals and persist to disk.
        FIX #3: Uses a file lock so only one gunicorn worker trains at a time.
        """
        lock_path = os.path.join(MODELS_DIR, f"{league_code}.lock")

        def _do_train():
            df = feature_service.build_training_dataframe(matches)

            if len(df) < settings.MIN_MATCHES_REQUIRED:
                raise ValueError(
                    f"Solo {len(df)} partidos válidos para entrenar {league_code}. "
                    f"Se necesitan al menos {settings.MIN_MATCHES_REQUIRED}."
                )

            # FIX #6: Use a dedicated holdout set (last 15%) for history accuracy metrics
            # Split chronologically — last 15% of matches as holdout (no shuffle)
            split_idx = int(len(df) * 0.85)
            df_train = df.iloc[:split_idx]
            df_holdout = df.iloc[split_idx:]

            X = df_train[FEATURE_COLS]
            y_home = df_train["target_home_goals"]
            y_away = df_train["target_away_goals"]

            X_train, X_val, y_h_train, y_h_val, y_a_train, y_a_val = train_test_split(
                X, y_home, y_away, test_size=0.15, random_state=42
            )

            model_params = {
                "n_estimators": 182,
                "max_depth": 3,
                "learning_rate": 0.011,
                "subsample": 0.728,
                "colsample_bytree": 0.835,
                "min_child_weight": 7,
                "random_state": 42,
                "verbosity": 0
            }

            try:
                from xgboost.callback import EarlyStopping
                es = EarlyStopping(rounds=20, save_best=True)
                xg_home = XGBRegressor(**model_params, callbacks=[es])
                xg_home.fit(X_train, y_h_train, eval_set=[(X_val, y_h_val)], verbose=False)

                xg_away = XGBRegressor(**model_params, callbacks=[es])
                xg_away.fit(X_train, y_a_train, eval_set=[(X_val, y_a_val)], verbose=False)
            except Exception:
                xg_home = XGBRegressor(**model_params)
                xg_home.fit(X_train, y_h_train, eval_set=[(X_val, y_h_val)], verbose=False)

                xg_away = XGBRegressor(**model_params)
                xg_away.fit(X_train, y_a_train, eval_set=[(X_val, y_a_val)], verbose=False)

            pred_h = xg_home.predict(X_val)
            pred_a = xg_away.predict(X_val)

            mae_h = mean_absolute_error(y_h_val, pred_h)
            rmse_h = np.sqrt(mean_squared_error(y_h_val, pred_h))
            mae_a = mean_absolute_error(y_a_val, pred_a)
            rmse_a = np.sqrt(mean_squared_error(y_a_val, pred_a))

            logger.info(f"[{league_code}] Home xG -> MAE: {mae_h:.3f}, RMSE: {rmse_h:.3f}")
            logger.info(f"[{league_code}] Away xG -> MAE: {mae_a:.3f}, RMSE: {rmse_a:.3f}")

            payload = {
                "xg_home": xg_home,
                "xg_away": xg_away,
                "n_rows": len(df_train),
                "mae_home": mae_h,
                "mae_away": mae_a,
                "trained_at": datetime.now().isoformat(),
                # Store holdout for unbiased history evaluation (FIX #6)
                "holdout_matches": df_holdout[["target_home_goals", "target_away_goals"]].to_dict(orient="records") if len(df_holdout) > 0 else []
            }

            # Probability Calibrator
            try:
                from app.services.poisson_service import poisson_service
                from sklearn.linear_model import LogisticRegression

                p_h_tr = np.maximum(0.01, xg_home.predict(X_train))
                p_a_tr = np.maximum(0.01, xg_away.predict(X_train))

                calib_X = []
                for h, a in zip(p_h_tr, p_a_tr):
                    mat = poisson_service.calculate_probability_matrix(h, a)
                    met = poisson_service.extract_metrics(mat)
                    calib_X.append([met["prob_away_win"] / 100, met["prob_draw"] / 100, met["prob_home_win"] / 100])

                calib_Y = []
                for h, a in zip(y_h_train, y_a_train):
                    if h > a: calib_Y.append(2)
                    elif h == a: calib_Y.append(1)
                    else: calib_Y.append(0)

                calibrator = LogisticRegression(max_iter=2000)
                calibrator.fit(calib_X, calib_Y)
                payload["calibrator"] = calibrator
                logger.info(f"[{league_code}] Probability Calibrator trained.")
            except Exception as e:
                logger.warning(f"Failed to train calibrator: {e}")

            joblib.dump(payload, self._model_path(league_code, timestamp=True))
            self._model_cache[league_code] = payload
            logger.info(f"[{league_code}] Model saved successfully.")
            return payload

        if _FILELOCK_AVAILABLE:
            lock = filelock.FileLock(lock_path, timeout=300)
            with lock:
                return _do_train()
        else:
            return _do_train()

    def ensure_model_ready(self, league_code: str, background_tasks=None):
        """Checks if model exists and is not stale. Triggers training if needed."""
        matches = data_service.get_historical_matches(league_code)
        payload = self._load_model(league_code)

        def trigger_training(matches_data):
            if self._training_in_progress.get(league_code):
                return
            self._training_in_progress[league_code] = True

            def train_task():
                try:
                    self._train_and_save(league_code, matches_data)
                except Exception as e:
                    logger.error(f"Error training {league_code}: {e}")
                finally:
                    self._training_in_progress[league_code] = False

            if background_tasks:
                background_tasks.add_task(train_task)
            else:
                train_task()

        if payload is None:
            if len(matches) < 10:
                return

            if self._training_in_progress.get(league_code):
                from fastapi import HTTPException
                raise HTTPException(status_code=202, detail={"status": "training", "message": "Entrenamiento en curso..."})

            trigger_training(matches)
            if background_tasks:
                from fastapi import HTTPException
                raise HTTPException(status_code=202, detail={"status": "training", "message": "Entrenamiento iniciado..."})

        elif payload.get("is_stale"):
            trigger_training(matches)

    def predict_xg(self, league_code: str, home_team_id: int, away_team_id: int, background_tasks=None) -> tuple:
        matches = data_service.get_historical_matches(league_code)
        payload = self._load_model(league_code)

        def trigger_training(matches_data):
            if self._training_in_progress.get(league_code):
                return
            logger.info(f"[{league_code}] Triggering background training...")
            self._training_in_progress[league_code] = True

            def train_task():
                try:
                    self._train_and_save(league_code, matches_data)
                except Exception as e:
                    logger.error(f"Error training model for {league_code}: {e}")
                finally:
                    self._training_in_progress[league_code] = False

            if background_tasks is not None:
                background_tasks.add_task(train_task)
            else:
                train_task()

        if payload is None:
            if len(matches) < 10:
                raise ValueError(
                    f"No hay suficientes partidos finalizados para {league_code}. "
                    "Intenta más tarde en la temporada."
                )

            if self._training_in_progress.get(league_code):
                from fastapi import HTTPException
                raise HTTPException(status_code=202, detail={"status": "training", "message": f"Modelo para {league_code} en entrenamiento."})

            trigger_training(matches)

            if background_tasks is not None:
                from fastapi import HTTPException
                raise HTTPException(status_code=202, detail={"status": "training", "message": f"Entrenamiento iniciado para {league_code}."})
            else:
                payload = self._load_model(league_code)

        elif payload.get("is_stale", False):
            logger.info(f"[{league_code}] Model stale ({payload.get('model_age_days')} days). Re-training.")
            trigger_training(matches)

        if payload is None:
            raise ValueError(f"No se pudo cargar ni entrenar el modelo para {league_code}.")

        xg_home_model: XGBRegressor = payload["xg_home"]
        xg_away_model: XGBRegressor = payload["xg_away"]

        feat_df = feature_service.build_prediction_features(home_team_id, away_team_id, matches)
        X_pred = feat_df[FEATURE_COLS]

        lambda_home = max(0.01, float(xg_home_model.predict(X_pred)[0]))
        lambda_away = max(0.01, float(xg_away_model.predict(X_pred)[0]))

        return lambda_home, lambda_away, payload


model_service = ModelService()
