"""
ML-05: Feedback Loop — Reconcilia predicciones pasadas con resultados reales.

Arquitectura:
─────────────
1. FeedbackService.run_for_league(league_code):
   - Lee predicciones sin resultado real desde Supabase
   - Busca el resultado real en los partidos históricos de football-data.org
   - Calcula Brier Score por predicción
   - Actualiza la fila en Supabase con el resultado y la métrica
   - Si el Brier Score promedio reciente > umbral → detecta model drift

2. Se llama desde el scheduler (cada 6h) para todas las ligas configuradas.

3. El Brier Score es la métrica estándar para evaluar modelos probabilísticos:
   BS = (p_home - actual_home)² + (p_draw - actual_draw)² + (p_away - actual_away)²
   Rango: [0.0, 2.0]. BS < 0.20 = modelo excelente, BS > 0.28 = drift probable.

Dependencias: ninguna nueva (usa supabase_service + data_service ya existentes).
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── Brier Score threshold for model drift detection ───────────────────────────
DRIFT_BRIER_THRESHOLD = 0.28   # above this → consider retraining
DRIFT_LOOKBACK_DAYS   = 14     # window for computing the rolling Brier Score
MIN_SAMPLES_FOR_DRIFT = 10     # minimum resolved predictions to trigger check


class FeedbackService:
    """
    Reconciles stored predictions with real match results and computes
    out-of-sample Brier Scores to monitor model performance over time.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def run_for_league(self, league_code: str) -> dict:
        """
        Main entry point — called by the scheduler once per cycle per league.

        Returns a summary dict with resolved count, avg Brier Score, and
        a drift_detected flag that the scheduler can act on.
        """
        from app.services.supabase_service import supabase_service
        from app.services.data_service import data_service

        if not supabase_service.enabled:
            logger.debug("[Feedback] Supabase disabled — skipping feedback loop.")
            return {"resolved": 0, "avg_brier": None, "drift_detected": False}

        # 1. Load finished matches from the API (or disk cache)
        finished = data_service.get_historical_matches(league_code)
        
        # Build lookup: match_id → result (we use (home_name, away_name, date) as key
        # since football-data doesn't expose match IDs directly in our stored format)
        result_lookup = self._build_result_lookup(finished)

        # 1.5 Live Feedback Loop: Enrich with instant results from ESPN
        # football-data is often slow to update finished matches. ESPN is real-time.
        espn_results = self._fetch_espn_recent_results(league_code)
        if espn_results:
            logger.info(f"[Feedback] [{league_code}] Injected {len(espn_results)} live results from ESPN.")
            result_lookup.update(espn_results)

        # 2. Fetch unresolved predictions from Supabase
        unresolved = supabase_service.get_unresolved_predictions(league_code)
        if not unresolved:
            logger.info(f"[Feedback] [{league_code}] No unresolved predictions to process.")
        else:
            logger.info(
                f"[Feedback] [{league_code}] Processing {len(unresolved)} unresolved predictions."
            )

        resolved_count = 0
        for pred in unresolved:
            result = self._match_result(pred, result_lookup)
            if result is None:
                continue   # Match hasn't been played yet

            brier = self._brier_score(
                pred["prob_home"], pred["prob_draw"], pred["prob_away"],
                result["actual_home"], result["actual_away"],
            )

            ok = supabase_service.resolve_prediction(
                match_id=pred["match_id"],
                actual_home=result["actual_home"],
                actual_away=result["actual_away"],
                actual_verdict=result["verdict"],
                brier_score=brier,
            )
            if ok:
                resolved_count += 1

        # 3. Compute rolling Brier Score for drift detection
        avg_brier, drift_detected = self._check_drift(league_code, supabase_service)

        if drift_detected:
            logger.warning(
                f"[Feedback] [{league_code}] MODEL DRIFT DETECTED — "
                f"avg Brier Score = {avg_brier:.4f} > {DRIFT_BRIER_THRESHOLD}. "
                "Scheduling immediate retraining."
            )
            self._trigger_drift_retrain(league_code)

        return {
            "resolved":       resolved_count,
            "avg_brier":      round(avg_brier, 4) if avg_brier is not None else None,
            "drift_detected": drift_detected,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_result_lookup(self, finished: list) -> dict:
        """
        Builds a lookup dict from finished matches for fast reconciliation.
        Key: (home_team_name_lower, away_team_name_lower, date_str_YYYY-MM-DD)
        Value: {"actual_home": int, "actual_away": int, "verdict": str}
        """
        lookup = {}
        if not finished:
            return lookup
            
        for m in finished:
            date_key = m["utcDate"][:10]   # YYYY-MM-DD
            key = (
                m["homeTeam_name"].lower().strip(),
                m["awayTeam_name"].lower().strip(),
                date_key,
            )
            h, a = int(m["homeGoals"]), int(m["awayGoals"])
            lookup[key] = {
                "actual_home": h,
                "actual_away": a,
                "verdict": "L" if h > a else ("V" if a > h else "E"),
            }
        return lookup

    def _fetch_espn_recent_results(self, league_code: str) -> dict:
        """
        Fetches the current ESPN scoreboard and returns a lookup dict of matches
        that have just completed (Full Time). This bypasses the delay in football-data.
        """
        import requests
        from app.services.market_service import ESPN_MAP
        
        espn_code = ESPN_MAP.get(league_code)
        if not espn_code:
            return {}
            
        lookup = {}
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return lookup
                
            data = r.json()
            for ev in data.get("events", []):
                # Only process matches that are actually finished
                status = ev.get("status", {}).get("type", {})
                if not status.get("completed", False):
                    continue
                    
                comp = ev.get("competitions", [])[0] if ev.get("competitions") else {}
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                    
                # ESPN standard: index 0 is usually home, index 1 is away, but let's check homeAway flag
                home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                
                h_name = home_c.get("team", {}).get("name", "").lower().strip()
                a_name = away_c.get("team", {}).get("name", "").lower().strip()
                
                h_score = int(home_c.get("score", 0))
                a_score = int(away_c.get("score", 0))
                
                # ESPN dates are in ISO format (e.g. 2026-05-01T20:00Z)
                date_str = ev.get("date", "")[:10]
                
                # Remove FC/FCs from ESPN names for better matching
                h_name = h_name.replace(" fc", "").replace("fc ", "")
                a_name = a_name.replace(" fc", "").replace("fc ", "")
                
                key = (h_name, a_name, date_str)
                lookup[key] = {
                    "actual_home": h_score,
                    "actual_away": a_score,
                    "verdict": "L" if h_score > a_score else ("V" if a_score > h_score else "E")
                }
                
        except Exception as e:
            logger.error(f"[Feedback] ESPN live score fetch failed for {league_code}: {e}")
            
        return lookup

    def _match_result(self, pred: dict, result_lookup: dict) -> dict | None:
        """Tries to find the real result for a stored prediction."""
        if not pred.get("utc_date"):
            return None

        date_key = str(pred["utc_date"])[:10]
        key = (
            pred["home_team"].lower().strip(),
            pred["away_team"].lower().strip(),
            date_key,
        )
        return result_lookup.get(key)

    @staticmethod
    def _brier_score(
        p_home: float, p_draw: float, p_away: float,
        actual_home: int, actual_away: int,
    ) -> float:
        """
        Multi-class Brier Score for a 3-outcome prediction (1X2).

        BS = Σ (p_i - o_i)²  where o_i ∈ {0, 1} is the one-hot actual outcome.
        Range: [0.0, 2.0].  Lower is better.
        A naive (33/33/33) prediction gives BS ≈ 0.667.
        A good model typically achieves BS ~ 0.20-0.24.
        """
        if actual_home > actual_away:
            o_home, o_draw, o_away = 1.0, 0.0, 0.0
        elif actual_away > actual_home:
            o_home, o_draw, o_away = 0.0, 0.0, 1.0
        else:
            o_home, o_draw, o_away = 0.0, 1.0, 0.0

        return (
            (p_home / 100 - o_home) ** 2
            + (p_draw / 100 - o_draw) ** 2
            + (p_away / 100 - o_away) ** 2
        )

    def _check_drift(self, league_code: str, supabase_service) -> tuple[float | None, bool]:
        """
        Computes the average Brier Score over the last DRIFT_LOOKBACK_DAYS for
        this league.  Returns (avg_brier, drift_detected).
        """
        try:
            recent = supabase_service.get_resolved_predictions(
                league_code, days=DRIFT_LOOKBACK_DAYS
            )
            if len(recent) < MIN_SAMPLES_FOR_DRIFT:
                return None, False

            scores = [r["brier_score"] for r in recent if r.get("brier_score") is not None]
            if not scores:
                return None, False

            avg = sum(scores) / len(scores)
            return avg, avg > DRIFT_BRIER_THRESHOLD

        except Exception as e:
            logger.error(f"[Feedback] [{league_code}] _check_drift error: {e}")
            return None, False

    def _trigger_drift_retrain(self, league_code: str) -> None:
        """Forces an immediate background retraining when drift is detected."""
        try:
            from app.services.model_service import model_service
            from app.services.data_service import data_service

            matches = data_service.get_historical_matches(league_code)
            model_service._trigger_background_training(league_code, matches)
            logger.info(
                f"[Feedback] [{league_code}] Drift-triggered retraining scheduled."
            )
        except Exception as e:
            logger.error(f"[Feedback] [{league_code}] Could not trigger retraining: {e}")


feedback_service = FeedbackService()
