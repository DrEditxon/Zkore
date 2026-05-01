import json
import os
import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

logger = logging.getLogger(__name__)

HYPERPARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "hyperparams.json")

class TuningService:
    def __init__(self):
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(HYPERPARAMS_FILE):
            try:
                with open(HYPERPARAMS_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[Tuning] Error cargando hyperparams: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(HYPERPARAMS_FILE, "w") as f:
                json.dump(self._cache, f, indent=4)
        except Exception as e:
            logger.error(f"[Tuning] Error guardando hyperparams: {e}")

    def get_best_params(self, league_code: str) -> dict:
        """Devuelve hiperparámetros cacheados para la liga o los default."""
        default_params = {
            "objective": "count:poisson",
            "n_estimators": 182,
            "max_depth": 3,
            "learning_rate": 0.011,
            "subsample": 0.728,
            "colsample_bytree": 0.835,
            "min_child_weight": 7,
            "random_state": 42,
            "verbosity": 0,
        }
        return self._cache.get(league_code, default_params)

    def tune_league(self, league_code: str, X_train, y_train):
        """
        ML-04 FIX: Búsqueda de hiperparámetros mejorada.

        Cambios respecto a la versión anterior:
        ─────────────────────────────────────────
        1. TimeSeriesSplit en lugar de KFold aleatorio.
           El KFold mezclaba partidos futuros en el training fold para validar
           partidos pasados → data leakage temporal.  TimeSeriesSplit garantiza
           que el fold de validación siempre es cronológicamente posterior al de
           entrenamiento.

        2. n_iter: 15 → 50.
           El espacio de búsqueda es 6-dimensional; 15 iteraciones son
           insuficientes para cubrirlo.  50 da 3x mejor cobertura.

        3. Scoring: neg_mean_absolute_error → neg_mean_squared_error.
           MSE penaliza errores grandes más que MAE, lo cual es más relevante
           para Poisson donde sobreestimar 3+ goles es muy costoso.

        4. Rangos ampliados para n_estimators, learning_rate y min_child_weight.
        """
        from sklearn.model_selection import TimeSeriesSplit

        logger.info(f"[{league_code}] Iniciando Tuning Automático (ML-04 mejorado)...")

        param_dist = {
            "n_estimators":      randint(50, 400),
            "max_depth":         randint(2, 7),
            "learning_rate":     uniform(0.005, 0.095),   # [0.005, 0.100]
            "subsample":         uniform(0.5, 0.45),       # [0.50, 0.95]
            "colsample_bytree":  uniform(0.5, 0.45),       # [0.50, 0.95]
            "min_child_weight":  randint(3, 20),
        }

        xgb = XGBRegressor(objective="count:poisson", random_state=42, verbosity=0)

        # ML-04 FIX: TimeSeriesSplit respects chronological order
        tscv = TimeSeriesSplit(n_splits=5)

        search = RandomizedSearchCV(
            xgb,
            param_distributions=param_dist,
            n_iter=50,                           # 3× more coverage than before
            cv=tscv,                             # temporal-aware cross-validation
            scoring="neg_mean_squared_error",    # penalises large errors more
            random_state=42,
            n_jobs=-1,
        )
        
        search.fit(X_train, y_train)
        
        best = search.best_params_
        best["objective"] = "count:poisson"
        best["random_state"] = 42
        best["verbosity"] = 0
        
        # Cast a tipos nativos para que sea serializable a JSON
        for k, v in best.items():
            if isinstance(v, np.integer):
                best[k] = int(v)
            elif isinstance(v, np.floating):
                best[k] = float(v)
                
        self._cache[league_code] = best
        self._save_cache()
        logger.info(f"[{league_code}] Tuning completado. Mejores params: {best}")
        
        return best

tuning_service = TuningService()
