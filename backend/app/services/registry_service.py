"""
Model Registry — Versioning y trazabilidad de modelos entrenados.

Persiste metadatos de cada entrenamiento en la tabla `model_registry` de Supabase:
  - version, accuracy, Brier Score, MAE, nº partidos, timestamp, liga

Permite:
  - Auditoría de degradación a lo largo del tiempo
  - Comparación entre versiones de modelos
  - Selección del mejor modelo por liga (si se expone a múltiples versiones)
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RegistryService:
    """Gestiona el ciclo de vida y los metadatos de los modelos entrenados."""

    def register_model(
        self,
        league_code: str,
        version: str,
        mae_home: float,
        mae_away: float,
        n_training: int,
        accuracy: float | None = None,
        brier_score: float | None = None,
        path: str | None = None,
    ) -> bool:
        """
        Registra un nuevo modelo entrenado en Supabase.
        Llamado desde model_service._train_and_save() después de cada entrenamiento.
        Devuelve True si se guardó con éxito.
        """
        from app.services.supabase_service import supabase_service
        if not supabase_service.enabled:
            logger.debug("[Registry] Supabase desactivado — registro local omitido.")
            return False

        record = {
            "league_code": league_code,
            "version":     version,
            "mae_home":    round(float(mae_home), 4),
            "mae_away":    round(float(mae_away), 4),
            "n_training":  int(n_training),
            "trained_at":  datetime.now(timezone.utc).isoformat(),
        }
        if accuracy is not None:
            record["accuracy"] = round(float(accuracy), 4)
        if brier_score is not None:
            record["brier_score"] = round(float(brier_score), 4)
        if path is not None:
            record["path"] = path

        try:
            endpoint = f"{supabase_service.url}/rest/v1/model_registry"
            r = supabase_service.session.post(
                endpoint,
                headers={
                    **supabase_service.headers,
                    "Prefer": "return=minimal",
                },
                json=record,
                timeout=8,
            )
            if r.status_code in (200, 201, 204):
                logger.info(
                    f"[Registry] [{league_code}] Registered model {version} "
                    f"(MAE: {mae_home:.3f}/{mae_away:.3f}, N={n_training})"
                )
                return True
            logger.warning(
                f"[Registry] [{league_code}] Insert failed: {r.status_code} {r.text[:200]}"
            )
        except Exception as e:
            logger.error(f"[Registry] [{league_code}] register_model error: {e}")

        return False

    def get_active_models(self, limit: int = 20) -> list[dict]:
        """
        Devuelve los modelos más recientes de cada liga, ordenados por fecha desc.
        Usado por el endpoint /models.
        """
        from app.services.supabase_service import supabase_service
        if not supabase_service.enabled:
            return self._get_active_models_local()

        try:
            endpoint = (
                f"{supabase_service.url}/rest/v1/model_registry"
                f"?order=trained_at.desc&limit={limit}"
            )
            r = supabase_service.session.get(endpoint, headers=supabase_service.headers, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"[Registry] get_active_models error: {e}")

        return self._get_active_models_local()

    def get_league_history(self, league_code: str, limit: int = 10) -> list[dict]:
        """Historial de versiones de modelo para una liga específica."""
        from app.services.supabase_service import supabase_service
        if not supabase_service.enabled:
            return []
        try:
            endpoint = (
                f"{supabase_service.url}/rest/v1/model_registry"
                f"?league_code=eq.{league_code}&order=trained_at.desc&limit={limit}"
            )
            r = supabase_service.session.get(endpoint, headers=supabase_service.headers, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"[Registry] get_league_history error: {e}")
        return []

    def _get_active_models_local(self) -> list[dict]:
        """
        Fallback local cuando Supabase no está disponible.
        Retorna datos desde el model_service cache en memoria.
        """
        from app.services.model_service import model_service
        from app.core.config import settings

        result = []
        for league_code in settings.LEAGUES_METADATA:
            payload = model_service._model_cache.get(league_code)
            if payload:
                result.append({
                    "league_code": league_code,
                    "version":     model_service.MODEL_VERSION,
                    "mae_home":    payload.get("mae_home"),
                    "mae_away":    payload.get("mae_away"),
                    "n_training":  payload.get("n_rows"),
                    "trained_at":  payload.get("trained_at"),
                    "accuracy":    None,
                    "brier_score": None,
                    "source":      "local_cache",
                })
        return result


registry_service = RegistryService()
