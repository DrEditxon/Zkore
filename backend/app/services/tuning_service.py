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
        Ejecuta RandomizedSearchCV para encontrar los hiperparámetros óptimos 
        (AutoML ligero) adaptados al estilo de goles de la liga.
        """
        logger.info(f"[{league_code}] Iniciando Tuning Automático de Hiperparámetros (Fase 3)...")
        
        param_dist = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 6),
            "learning_rate": uniform(0.005, 0.05),
            "subsample": uniform(0.5, 0.4),
            "colsample_bytree": uniform(0.5, 0.4),
            "min_child_weight": randint(3, 15),
        }
        
        xgb = XGBRegressor(objective="count:poisson", random_state=42, verbosity=0)
        
        # Búsqueda aleatoria: 15 iteraciones con validación cruzada k=3
        search = RandomizedSearchCV(
            xgb, param_distributions=param_dist,
            n_iter=15, cv=3, scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=-1
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
