import os
import glob
import logging
import joblib

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime

from app.core.config import settings
from app.services.data_service import data_service
from app.services.feature_service import feature_service, FEATURE_COLS
from app.services.tuning_service import tuning_service

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# FIX #3: File-based lock — prevents multiple gunicorn workers training simultaneously
try:
    import filelock
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False
    logger.warning("filelock not installed. Run: pip install filelock")


class ModelService:
    def __init__(self):
        self._model_cache: dict = {}
        self._training_in_progress: dict = {}

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _model_path(self, league_code: str, timestamp: bool = True) -> str:
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(MODELS_DIR, f"{league_code}_xgb_{ts}.joblib")
        return os.path.join(MODELS_DIR, f"{league_code}_xgb_*.joblib")

    # ── Load ─────────────────────────────────────────────────────────────────

    def _load_model(self, league_code: str):
        payload = self._model_cache.get(league_code)

        # 1. Try local disk (fast)
        if payload is None:
            files = glob.glob(self._model_path(league_code, timestamp=False))
            if files:
                latest = max(files, key=lambda f: os.path.basename(f))
                try:
                    loaded = joblib.load(latest)
                    if "xg_home" in loaded:
                        payload = loaded
                        self._model_cache[league_code] = payload
                        logger.info(f"[{league_code}] Loaded model from disk: {os.path.basename(latest)}")
                except Exception as e:
                    logger.error(f"[{league_code}] Failed to load model from disk {latest}: {e}")

        # 2. Fallback: download from Supabase Storage
        if payload is None:
            payload = self._load_model_from_supabase(league_code)

        if payload:
            trained_at_raw = payload.get("trained_at")
            if trained_at_raw:
                trained_at = datetime.fromisoformat(trained_at_raw)
                age_days = (datetime.now() - trained_at).total_seconds() / 86400.0
            else:
                age_days = 999.0

            payload["model_age_days"] = round(age_days, 2)
            payload["is_stale"] = age_days > 7.0

        return payload

    def _load_model_from_supabase(self, league_code: str):
        """
        Downloads and deserializes a model from Supabase Storage.
        Also caches it to local disk so the next process start is instant.
        """
        try:
            from app.services.supabase_service import supabase_service
            model_bytes = supabase_service.download_model(league_code)
            if model_bytes is None:
                return None
            buf = __import__('io').BytesIO(model_bytes)
            loaded = joblib.load(buf)
            if "xg_home" not in loaded:
                return None
            # Cache in memory
            self._model_cache[league_code] = loaded
            # Cache on disk so next startup is instant
            local_path = self._model_path(league_code, timestamp=True)
            joblib.dump(loaded, local_path)
            logger.info(f"[{league_code}] Model downloaded from Supabase and cached locally.")
            return loaded
        except Exception as e:
            logger.error(f"[{league_code}] _load_model_from_supabase error: {e}")
            return None

    # ── Cleanup old model files ───────────────────────────────────────────────

    def _cleanup_old_models(self, league_code: str, keep: int = 2):
        """
        BUG A FIX: Delete old .joblib files after retraining.
        Keeps the `keep` most recent files as a safety backup.
        """
        files = sorted(
            glob.glob(self._model_path(league_code, timestamp=False)),
            key=lambda f: os.path.basename(f),
            reverse=True,
        )
        for old_file in files[keep:]:
            try:
                os.remove(old_file)
                logger.info(f"[{league_code}] Deleted old model: {os.path.basename(old_file)}")
            except Exception as e:
                logger.warning(f"[{league_code}] Could not delete {old_file}: {e}")

    # ── Cross-process training guard + background trigger ──────────────────────

    def _is_training_locked(self, league_code: str) -> bool:
        """
        BUG-03 FIX: Cross-process-safe training guard.

        The in-memory dict `_training_in_progress` only works within a single
        Gunicorn worker.  With N workers each process has its own dict, so a
        concurrent training attempt in another worker is invisible here.

        This method probes the filelock with timeout=0 (non-blocking):
          - Lock acquired instantly  →  nobody is training  →  return False
          - Timeout (lock held)      →  some worker is training  →  return True

        Falls back to the in-memory dict when filelock is not available.
        """
        if not _FILELOCK_AVAILABLE:
            return self._training_in_progress.get(league_code, False)

        lock_path = os.path.join(MODELS_DIR, f"{league_code}.lock")
        try:
            probe = filelock.FileLock(lock_path, timeout=0)
            probe.acquire()
            probe.release()
            return False   # Successfully acquired → no training in progress
        except filelock.Timeout:
            return True    # Lock is held by some worker → training in progress

    def _trigger_background_training(
        self,
        league_code: str,
        matches_data: list,
        background_tasks=None,
    ) -> None:
        """
        BUG-03 FIX (cont.): Single authoritative method to schedule background
        training, replacing the two duplicated inner trigger_training() closures
        that previously existed in ensure_model_ready() and predict_xg().

        Uses _is_training_locked() for cross-process safety before setting the
        in-memory flag and spinning up the worker thread / FastAPI background task.
        """
        if self._is_training_locked(league_code):
            logger.info(
                f"[{league_code}] Training already locked (cross-process check). Skipping."
            )
            return

        # Set in-memory flag for THIS worker to short-circuit duplicate calls
        self._training_in_progress[league_code] = True

        def train_task():
            try:
                self._train_and_save(league_code, matches_data)
            except Exception as e:
                logger.error(f"[{league_code}] Background training error: {e}")
            finally:
                self._training_in_progress[league_code] = False

        if background_tasks is not None:
            background_tasks.add_task(train_task)
        else:
            import threading
            threading.Thread(target=train_task, daemon=True).start()

    # ── Train ─────────────────────────────────────────────────────────────────

    def _train_and_save(self, league_code: str, matches: list):
        """
        Train XGBRegressors for expected goals and persist to disk.

        BUG A FIX: Cleans up old .joblib files after saving the new model.
        BUG B FIX: Invalidates historical data cache before training so the
                   model always trains on the freshest available data.
        BUG C FIX: Invalidates feature_service state cache so the next
                   prediction recomputes team ELO / form from scratch.
        FIX #3:    Uses a file lock so only one gunicorn worker trains at a time.
        """
        lock_path = os.path.join(MODELS_DIR, f"{league_code}.lock")

        def _do_train():
            # ML-01 FIX: Use multi-season data for training.
            # After the historical bootstrap completes, this returns matches from
            # up to 3 past seasons merged with the current live season, giving
            # the model 3-4x more training data and better generalisation.
            # Falls back to current-season-only if Supabase is not available.
            logger.info(f"[{league_code}] Invalidating caches before retraining...")
            data_service.invalidate_cache(league_code)          # disk + memory cache
            fresh_matches = data_service.get_historical_matches_multi_season(league_code)

            # BUG C FIX: Flush feature state cache so next prediction uses fresh ELO
            feature_service.invalidate_cache()

            df = feature_service.build_training_dataframe(fresh_matches)

            if len(df) < settings.MIN_MATCHES_REQUIRED:
                raise ValueError(
                    f"Solo {len(df)} partidos válidos para entrenar {league_code}. "
                    f"Se necesitan al menos {settings.MIN_MATCHES_REQUIRED}."
                )

            # Chronological holdout — last 15% never seen during training
            split_idx = int(len(df) * 0.85)
            df_train  = df.iloc[:split_idx]
            df_holdout = df.iloc[split_idx:]

            X      = df_train[FEATURE_COLS]
            y_home = df_train["target_home_goals"]
            y_away = df_train["target_away_goals"]

            # PHASE 1: Ponderación temporal (Sample Weights)
            # Los partidos más recientes tienen mayor peso (decaimiento exponencial)
            decay_factor = 2.0
            weights = np.exp(np.linspace(-decay_factor, 0, len(X)))

            X_train, X_val, y_h_train, y_h_val, y_a_train, y_a_val, w_train, w_val = train_test_split(
                X, y_home, y_away, weights, test_size=0.15, random_state=42
            )

            # FASE 3: Obtener mejores hiperparámetros (si no existen, los calcula vía AutoML)
            model_params = tuning_service.get_best_params(league_code)
            
            # Si detecta los parámetros default, fuerza el tuning una sola vez
            if model_params.get("n_estimators") == 182:
                model_params = tuning_service.tune_league(league_code, X_train, y_h_train)

            try:
                from xgboost.callback import EarlyStopping
                es = EarlyStopping(rounds=20, save_best=True)
                xg_home = XGBRegressor(**model_params, callbacks=[es])
                xg_home.fit(X_train, y_h_train, eval_set=[(X_val, y_h_val)], sample_weight=w_train, verbose=False)
                xg_away = XGBRegressor(**model_params, callbacks=[es])
                xg_away.fit(X_train, y_a_train, eval_set=[(X_val, y_a_val)], sample_weight=w_train, verbose=False)
            except Exception:
                xg_home = XGBRegressor(**model_params)
                xg_home.fit(X_train, y_h_train, eval_set=[(X_val, y_h_val)], sample_weight=w_train, verbose=False)
                xg_away = XGBRegressor(**model_params)
                xg_away.fit(X_train, y_a_train, eval_set=[(X_val, y_a_val)], sample_weight=w_train, verbose=False)

            mae_h  = mean_absolute_error(y_h_val, xg_home.predict(X_val))
            rmse_h = float(np.sqrt(mean_squared_error(y_h_val, xg_home.predict(X_val))))
            mae_a  = mean_absolute_error(y_a_val, xg_away.predict(X_val))
            rmse_a = float(np.sqrt(mean_squared_error(y_a_val, xg_away.predict(X_val))))

            logger.info(f"[{league_code}] Home xG → MAE: {mae_h:.3f}  RMSE: {rmse_h:.3f}")
            logger.info(f"[{league_code}] Away xG → MAE: {mae_a:.3f}  RMSE: {rmse_a:.3f}")

            # ML-03 FIX: Isotonic Regression calibration.
            # XGBoost λ values are systematically biased. IsotonicRegression
            # learns the monotonic mapping λ_pred -> λ_actual using X_val
            # (which XGBoost never trained on). out_of_bounds='clip' prevents
            # extrapolation beyond the calibration range.
            from sklearn.isotonic import IsotonicRegression

            lhv = np.maximum(0.01, xg_home.predict(X_val).astype(float))
            lav = np.maximum(0.01, xg_away.predict(X_val).astype(float))

            calibrator_home = IsotonicRegression(out_of_bounds="clip", increasing=True)
            calibrator_home.fit(lhv, y_h_val.astype(float))

            calibrator_away = IsotonicRegression(out_of_bounds="clip", increasing=True)
            calibrator_away.fit(lav, y_a_val.astype(float))

            mae_h_cal = mean_absolute_error(y_h_val, calibrator_home.predict(lhv))
            mae_a_cal = mean_absolute_error(y_a_val, calibrator_away.predict(lav))
            logger.info(
                f"[{league_code}] Calibration: "
                f"home {mae_h:.3f}->{mae_h_cal:.3f} | away {mae_a:.3f}->{mae_a_cal:.3f}"
            )

            # BUG-01 FIX: Save rich holdout metadata for true OOS evaluation.
            _META_COLS = [
                "_match_home_id", "_match_away_id",
                "_match_home_name", "_match_away_name", "_match_date",
                "target_home_goals", "target_away_goals",
            ]
            available_meta = [c for c in _META_COLS if c in df_holdout.columns]

            payload = {
                "xg_home":          xg_home,
                "xg_away":          xg_away,
                "calibrator_home":  calibrator_home,  # ML-03
                "calibrator_away":  calibrator_away,
                "n_rows":           len(df_train),
                "mae_home":         mae_h,
                "mae_away":         mae_a,
                "mae_home_cal":     mae_h_cal,
                "mae_away_cal":     mae_a_cal,
                "trained_at":       datetime.now().isoformat(),
                "holdout_matches": (
                    df_holdout[available_meta].to_dict(orient="records")
                    if len(df_holdout) > 0 and available_meta else []
                ),
            }

            # Save new model to local disk
            new_path = self._model_path(league_code, timestamp=True)
            joblib.dump(payload, new_path)
            logger.info(f"[{league_code}] New model saved to disk: {os.path.basename(new_path)}")

            # Upload to Supabase Storage (persistent across deploys)
            try:
                import io
                from app.services.supabase_service import supabase_service
                buf = io.BytesIO()
                joblib.dump(payload, buf)
                buf.seek(0)
                supabase_service.upload_model(league_code, buf.read())
            except Exception as upload_err:
                logger.warning(f"[{league_code}] Supabase upload failed (non-fatal): {upload_err}")

            # Update in-memory cache immediately
            self._model_cache[league_code] = payload

            # BUG A FIX: Remove old model files (keep 2 as backup)
            self._cleanup_old_models(league_code, keep=2)

            return payload

        if _FILELOCK_AVAILABLE:
            lock = filelock.FileLock(lock_path, timeout=300)
            with lock:
                return _do_train()
        else:
            return _do_train()

    # ── ensure_model_ready ────────────────────────────────────────────────────

    def ensure_model_ready(self, league_code: str, background_tasks=None):
        """
        Called from the /upcoming endpoint.
        - No model → start training in background, return 202.
        - Model is stale → retrain in background, serve stale predictions silently.
        - Model is fresh → do nothing.

        BUG-03 FIX: Uses _is_training_locked() (cross-process) + the shared
        _trigger_background_training() method instead of the old per-method
        inner function with an in-memory-only guard.
        """
        matches = data_service.get_historical_matches(league_code)
        payload = self._load_model(league_code)

        if payload is None:
            if len(matches) < 10:
                return   # Not enough data yet — fail silently

            if self._is_training_locked(league_code):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=202,
                    detail={"status": "training", "message": "Entrenamiento en curso..."}
                )

            self._trigger_background_training(league_code, matches, background_tasks)
            if background_tasks:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=202,
                    detail={"status": "training", "message": "Entrenamiento iniciado..."}
                )

        elif payload.get("is_stale"):
            age = payload.get("model_age_days", "?")
            logger.info(f"[{league_code}] Model is {age} days old — scheduling background retraining.")
            self._trigger_background_training(league_code, matches, background_tasks)
            # Do NOT raise 202 here — continue serving stale predictions transparently

    # ── predict_xg ────────────────────────────────────────────────────────────

    def predict_xg(
        self,
        league_code: str,
        home_team_id: int,
        away_team_id: int,
        background_tasks=None,
    ) -> tuple:
        """
        BUG-03 FIX: Uses _is_training_locked() + _trigger_background_training()
        instead of the old duplicated inner trigger_training() closure.
        """
        matches = data_service.get_historical_matches(league_code)
        payload = self._load_model(league_code)

        if payload is None:
            if len(matches) < 10:
                raise ValueError(
                    f"No hay suficientes partidos finalizados para {league_code}."
                )

            if self._is_training_locked(league_code):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=202,
                    detail={"status": "training", "message": f"Modelo para {league_code} entrenándose."}
                )

            self._trigger_background_training(league_code, matches, background_tasks)

            if background_tasks is not None:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=202,
                    detail={"status": "training", "message": f"Entrenamiento iniciado para {league_code}."}
                )
            else:
                payload = self._load_model(league_code)

        elif payload.get("is_stale", False):
            logger.info(
                f"[{league_code}] Stale model ({payload.get('model_age_days')} days) "
                "— retraining in background."
            )
            self._trigger_background_training(league_code, matches, background_tasks)
            # Continue prediction with the stale model — no user-facing 202

        if payload is None:
            raise ValueError(f"No se pudo cargar ni entrenar el modelo para {league_code}.")

        feat_df = feature_service.build_prediction_features(home_team_id, away_team_id, matches)
        X_pred  = feat_df[FEATURE_COLS]

        lambda_home = max(0.01, float(payload["xg_home"].predict(X_pred)[0]))
        lambda_away = max(0.01, float(payload["xg_away"].predict(X_pred)[0]))

        # ML-03: Apply isotonic calibrators if present (backward compatible).
        cal_h = payload.get("calibrator_home")
        cal_a = payload.get("calibrator_away")
        if cal_h is not None:
            lambda_home = max(0.01, float(cal_h.predict([lambda_home])[0]))
        if cal_a is not None:
            lambda_away = max(0.01, float(cal_a.predict([lambda_away])[0]))

        return lambda_home, lambda_away, payload

    # ── Migration ─────────────────────────────────────────────────────────────

    def migrate_local_models_to_supabase(self):
        """
        One-time migration: uploads the latest local .joblib for each league
        to Supabase Storage if it isn't already there.
        Safe to call on every startup — skips leagues that already exist in the bucket.
        If the Supabase key changes, missing models are automatically re-uploaded.
        """
        from app.services.supabase_service import supabase_service
        if not supabase_service.enabled:
            return

        import io

        # Discover which leagues have local models
        all_files = glob.glob(os.path.join(MODELS_DIR, "*_xgb_*.joblib"))
        leagues = set()
        for f in all_files:
            basename = os.path.basename(f)
            parts = basename.split("_xgb_")
            if parts:
                leagues.add(parts[0])

        if not leagues:
            logger.info("[Migration] No local models found to migrate.")
            return

        # Check what's already in Supabase Storage
        existing = {item["name"] for item in supabase_service.list_models()}
        logger.info(f"[Migration] Models already in Supabase: {existing or 'none'}")

        for league_code in sorted(leagues):
            filename = f"{league_code}_xgb_latest.joblib"
            if filename in existing:
                logger.info(f"[Migration] [{league_code}] Already in Supabase — skipping.")
                continue

            # Find the latest local file for this league
            files = sorted(
                glob.glob(os.path.join(MODELS_DIR, f"{league_code}_xgb_*.joblib")),
                key=os.path.basename,
                reverse=True,
            )
            if not files:
                continue

            try:
                loaded = joblib.load(files[0])
                if "xg_home" not in loaded:
                    continue
                buf = io.BytesIO()
                joblib.dump(loaded, buf)
                buf.seek(0)
                ok = supabase_service.upload_model(league_code, buf.read())
                if ok:
                    logger.info(f"[Migration] [{league_code}] Migrated to Supabase Storage.")
            except Exception as e:
                logger.error(f"[Migration] [{league_code}] Failed: {e}")


model_service = ModelService()
