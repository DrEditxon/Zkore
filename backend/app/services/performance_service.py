"""
Performance Service — ROI real, accuracy, hit rate y calibración del modelo.

Calcula métricas reales sobre predicciones resueltas (con resultado real) almacenadas
en Supabase. Integra con el feedback loop (Brier Score) para dar una visión completa
del rendimiento del sistema de apuestas.

Métricas calculadas:
  - accuracy       : % de predicciones con verdict correcto
  - hit_rate       : igual que accuracy (alias estándar en betting)
  - roi            : Return on Investment simulado con Kelly fraccionado
  - avg_brier      : Brier Score promedio (calibración probabilística)
  - profit         : profit/loss acumulado sobre 1 unidad de stake
  - n_resolved     : nº de predicciones con resultado real
  - by_league      : métricas desglosadas por liga
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── Configuración de simulación de apuestas ───────────────────────────────────
SIMULATED_ODDS = {
    "L": 2.10,   # Odds promedio local (cuota típica bookie)
    "E": 3.40,   # Odds promedio empate
    "V": 2.60,   # Odds promedio visitante
}
STAKE_PER_BET = 1.0   # 1 unidad por apuesta (para calcular ROI)


class PerformanceService:
    """
    Calcula métricas de rendimiento reales sobre predicciones históricas
    resueltas. Funciona con o sin Supabase (fallback a holdout local).
    """

    def get_global_performance(self, days: int = 30) -> dict:
        """
        Métricas globales de todas las ligas en los últimos `days` días.
        Endpoint: GET /performance
        """
        from app.services.supabase_service import supabase_service

        if not supabase_service.enabled:
            return self._get_performance_local()

        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            endpoint = (
                f"{supabase_service.url}/rest/v1/predictions"
                f"?utc_date=gte.{cutoff}"
                f"&actual_verdict=not.is.null"
                f"&order=utc_date.desc"
                f"&limit=2000"
            )
            r = supabase_service.session.get(endpoint, headers=supabase_service.headers, timeout=10)
            if r.status_code != 200:
                return self._get_performance_local()

            records = r.json()
            if not records:
                return self._empty_performance()

            return self._compute_metrics(records)

        except Exception as e:
            logger.error(f"[Performance] get_global_performance error: {e}")
            return self._get_performance_local()

    def get_league_performance(self, league_code: str, days: int = 30) -> dict:
        """
        Métricas de una liga específica en los últimos `days` días.
        Endpoint: GET /performance/{league_code}
        """
        from app.services.supabase_service import supabase_service

        if not supabase_service.enabled:
            return self._get_performance_local(league_code)

        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            endpoint = (
                f"{supabase_service.url}/rest/v1/predictions"
                f"?league_code=eq.{league_code}"
                f"&utc_date=gte.{cutoff}"
                f"&actual_verdict=not.is.null"
                f"&order=utc_date.desc"
                f"&limit=500"
            )
            r = supabase_service.session.get(endpoint, headers=supabase_service.headers, timeout=10)
            if r.status_code != 200:
                return self._get_performance_local(league_code)

            records = r.json()
            if not records:
                return self._empty_performance()

            return self._compute_metrics(records)

        except Exception as e:
            logger.error(f"[Performance] get_league_performance error: {e}")
            return self._get_performance_local(league_code)

    # ── Core metric computation ───────────────────────────────────────────────

    def _compute_metrics(self, records: list) -> dict:
        """
        Calcula todas las métricas sobre una lista de predicciones resueltas.
        Cada record debe tener: verdict, actual_verdict, prob_home, prob_draw,
        prob_away, brier_score (nullable), league_code.
        """
        n = len(records)
        if n == 0:
            return self._empty_performance()

        hits = 0
        roi_cumulative = 0.0
        brier_scores = []
        by_league: dict = {}

        for rec in records:
            verdict = rec.get("verdict")           # Predicción del modelo: L/E/V
            actual  = rec.get("actual_verdict")    # Resultado real: L/E/V
            league  = rec.get("league_code", "??")

            # ── Hit Rate ──────────────────────────────────────────────────────
            is_hit = (verdict == actual)
            if is_hit:
                hits += 1

            # ── ROI simulado ──────────────────────────────────────────────────
            # Asumimos 1 unidad de stake en la predicción del modelo
            if verdict:
                odds = SIMULATED_ODDS.get(verdict, 2.0)
                if is_hit:
                    roi_cumulative += (odds - 1.0) * STAKE_PER_BET
                else:
                    roi_cumulative -= STAKE_PER_BET

            # ── Brier Score ───────────────────────────────────────────────────
            bs = rec.get("brier_score")
            if bs is not None:
                brier_scores.append(float(bs))

            # ── By league ─────────────────────────────────────────────────────
            if league not in by_league:
                by_league[league] = {"n": 0, "hits": 0, "roi": 0.0}
            by_league[league]["n"]    += 1
            by_league[league]["hits"] += 1 if is_hit else 0
            by_league[league]["roi"]  += (odds - 1.0) * STAKE_PER_BET if is_hit else -STAKE_PER_BET

        accuracy  = round((hits / n) * 100, 1)
        roi_pct   = round((roi_cumulative / (n * STAKE_PER_BET)) * 100, 1)
        avg_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None

        # Compute per-league summary
        league_summary = []
        for code, stats in sorted(by_league.items(), key=lambda x: -x[1]["n"]):
            n_l = stats["n"]
            league_summary.append({
                "league_code": code,
                "n":           n_l,
                "accuracy":    round((stats["hits"] / n_l) * 100, 1),
                "roi":         round((stats["roi"] / (n_l * STAKE_PER_BET)) * 100, 1),
            })

        return {
            "n_resolved":  n,
            "hits":        hits,
            "misses":      n - hits,
            "accuracy":    accuracy,
            "hit_rate":    accuracy,   # alias
            "roi":         roi_pct,
            "profit":      round(roi_cumulative, 2),
            "avg_brier":   avg_brier,
            "by_league":   league_summary,
            "period_days": None,  # set externally
            "simulated_odds": SIMULATED_ODDS,
        }

    # ── Local fallback (no Supabase) ──────────────────────────────────────────

    def _get_performance_local(self, league_code: str | None = None) -> dict:
        """
        Fallback cuando Supabase no está disponible.
        Calcula métricas usando el holdout set del modelo en memoria.
        """
        from app.services.model_service import model_service
        from app.core.config import settings
        from app.core.pipeline import predict_match
        from fastapi import HTTPException

        leagues = [league_code] if league_code else list(settings.LEAGUES_METADATA.keys())
        all_records = []

        for lc in leagues:
            payload = model_service._model_cache.get(lc)
            if not payload:
                continue

            holdout = payload.get("holdout_matches", [])
            for h in holdout[-5:]:  # Reducido de 20 a 5 para mejorar velocidad de carga
                try:
                    pred = predict_match(
                        lc,
                        int(h["_match_home_id"]),
                        int(h["_match_away_id"]),
                        h["_match_home_name"],
                        h["_match_away_name"],
                        utc_date=h["_match_date"],
                    )
                    probs = pred["probabilidades"]
                    act_h = int(h["target_home_goals"])
                    act_a = int(h["target_away_goals"])

                    if act_h > act_a:   actual = "L"
                    elif act_a > act_h: actual = "V"
                    else:               actual = "E"

                    if probs["local"] >= probs["empate"] and probs["local"] >= probs["visitante"]:
                        verdict = "L"
                    elif probs["visitante"] > probs["local"] and probs["visitante"] > probs["empate"]:
                        verdict = "V"
                    else:
                        verdict = "E"

                    # Brier score local
                    if act_h > act_a:   oh, od, oa = 1, 0, 0
                    elif act_a > act_h: oh, od, oa = 0, 0, 1
                    else:               oh, od, oa = 0, 1, 0

                    bs = (
                        (probs["local"]/100 - oh)**2 +
                        (probs["empate"]/100 - od)**2 +
                        (probs["visitante"]/100 - oa)**2
                    )

                    all_records.append({
                        "verdict":        verdict,
                        "actual_verdict": actual,
                        "prob_home":      probs["local"],
                        "prob_draw":      probs["empate"],
                        "prob_away":      probs["visitante"],
                        "brier_score":    round(bs, 4),
                        "league_code":    lc,
                    })
                except (HTTPException, Exception):
                    continue

        if not all_records:
            return self._empty_performance()

        result = self._compute_metrics(all_records)
        result["source"] = "local_holdout"
        return result

    @staticmethod
    def _empty_performance() -> dict:
        return {
            "n_resolved":  0,
            "hits":        0,
            "misses":      0,
            "accuracy":    0.0,
            "hit_rate":    0.0,
            "roi":         0.0,
            "profit":      0.0,
            "avg_brier":   None,
            "by_league":   [],
            "simulated_odds": SIMULATED_ODDS,
            "note": "Sin suficientes predicciones resueltas aún.",
        }


performance_service = PerformanceService()
