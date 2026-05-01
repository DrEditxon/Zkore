import logging
import concurrent.futures
from app.services.data_service import data_service
from app.core.pipeline import predict_match

logger = logging.getLogger(__name__)

# Columns present in the NEW holdout format (set by model_service after BUG-01 FIX)
_HOLDOUT_META_COLS = {
    "_match_home_id", "_match_away_id",
    "_match_home_name", "_match_away_name", "_match_date",
}


class HistoryService:
    def get_league_history(self, league_code: str, limit: int = 10):
        """
        Returns the last N predictions evaluated against REAL results.

        BUG-01 FIX — Data Leakage elimination:
        The previous implementation ran predict_match() on the last N FINISHED
        matches, which are the SAME matches the model was trained on. This
        inflated the reported accuracy (data leakage).

        New strategy (priority order):
        1. Use the model's own holdout set — the chronological last 15% of
           matches that were NEVER seen during training → true out-of-sample.
        2. Fallback: if the model is stale / pre-fix (no metadata in holdout),
           fall back to the legacy behaviour and log a warning.
        """
        from app.services.model_service import model_service
        from fastapi import HTTPException

        try:
            payload = model_service._load_model(league_code)
        except Exception as e:
            logger.error(f"[History] Could not load model for {league_code}: {e}")
            payload = None

        # ── Path 1: Rich holdout available (BUG-01 FIX) ──────────────────────
        if payload:
            holdout = payload.get("holdout_matches", [])
            has_meta = holdout and _HOLDOUT_META_COLS.issubset(holdout[0].keys())

            if has_meta:
                return self._evaluate_holdout(league_code, holdout, limit, payload)

            logger.warning(
                f"[{league_code}] Holdout has no match metadata — model was "
                "trained before BUG-01 fix. Falling back to legacy history. "
                "Accuracy will be inflated until the model is retrained."
            )

        # ── Path 2: Legacy fallback (original behaviour) ──────────────────────
        return self._evaluate_recent_matches_legacy(league_code, limit)

    # ── True out-of-sample evaluation ─────────────────────────────────────────

    def _evaluate_holdout(
        self,
        league_code: str,
        holdout: list,
        limit: int,
        payload: dict,
    ) -> dict:
        """
        Evaluates the model on holdout matches — the chronological last 15% of
        the training dataset that were NEVER used during model fitting.

        Returns accurate, unbiased accuracy metrics.
        """
        # Take the last `limit` holdout matches (most recent)
        recent_holdout = holdout[-limit:]

        def process_holdout_match(h):
            try:
                home_id   = int(h["_match_home_id"])
                away_id   = int(h["_match_away_id"])
                home_name = h["_match_home_name"]
                away_name = h["_match_away_name"]
                match_date = h["_match_date"]

                actual_home = int(h["target_home_goals"])
                actual_away = int(h["target_away_goals"])

                if actual_home > actual_away:   actual_verdict = "L"
                elif actual_away > actual_home: actual_verdict = "V"
                else:                           actual_verdict = "E"

                # Predict — model_service uses the loaded model directly
                prediction = predict_match(
                    league_code, home_id, away_id, home_name, away_name,
                    utc_date=match_date,
                )

                probs = prediction["probabilidades"]
                if probs["local"] > probs["empate"] and probs["local"] > probs["visitante"]:
                    pred_verdict = "L"
                elif probs["visitante"] > probs["empate"] and probs["visitante"] > probs["local"]:
                    pred_verdict = "V"
                else:
                    pred_verdict = "E"

                winner_hit = (actual_verdict == pred_verdict)

                predicted_total = (
                    prediction["expected_goals"]["local"]
                    + prediction["expected_goals"]["visitante"]
                )
                actual_total = actual_home + actual_away
                goals_hit = abs(predicted_total - actual_total) <= 1.0

                return {
                    "match":        f"{home_name} vs {away_name}",
                    "date":         match_date,
                    "actual_score": f"{actual_home}-{actual_away}",
                    "prediction": {
                        "winner": pred_verdict,
                        "probs":  probs,
                        "xg":     prediction["expected_goals"],
                    },
                    "is_hit": winner_hit,
                    "details": {
                        "winner_hit": winner_hit,
                        "goals_hit":  goals_hit,
                    },
                    "source": "holdout",  # UI hint: these are true OOS results
                }
            except Exception as e:
                from fastapi import HTTPException
                if isinstance(e, HTTPException):
                    raise e
                logger.error(f"[History] Holdout match processing error: {e}")
                return None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_holdout_match, recent_holdout))

            history_results = [r for r in results if r is not None]
            hits   = sum(1 for r in history_results if r["is_hit"])
            misses = len(history_results) - hits

            return {
                "summary": {
                    "hits":     hits,
                    "misses":   misses,
                    "total":    len(history_results),
                    "accuracy": round((hits / len(history_results)) * 100, 1) if history_results else 0,
                    "source":   "holdout",   # True out-of-sample — no data leakage
                    "note":     "Precisión calculada sobre el conjunto holdout (15% más reciente, nunca visto en entrenamiento).",
                },
                "history": history_results,
            }

        except Exception as e:
            from fastapi import HTTPException
            if isinstance(e, HTTPException) and e.status_code == 202:
                return {
                    "training_in_progress": True,
                    "summary": {"hits": 0, "misses": 0, "total": 0, "accuracy": 0},
                    "history": [],
                }
            raise

    # ── Legacy fallback (pre-fix behaviour) ───────────────────────────────────

    def _evaluate_recent_matches_legacy(self, league_code: str, limit: int) -> dict:
        """
        Original implementation kept for backward compatibility.
        ⚠️  Contains DATA LEAKAGE — accuracy is inflated because the model
        was trained on these same matches. Used only while the model hasn't
        been retrained with the BUG-01 fix yet.
        """
        from fastapi import HTTPException

        matches_raw = data_service.get_historical_matches(league_code)
        matches_raw.sort(key=lambda x: x["utcDate"], reverse=True)
        recent_matches = matches_raw[:limit]

        history_results = []

        def process_match(m):
            try:
                prediction = predict_match(
                    league_code,
                    m["homeTeam_id"],
                    m["awayTeam_id"],
                    m["homeTeam_name"],
                    m["awayTeam_name"],
                    utc_date=m["utcDate"],
                )

                actual_home = m["homeGoals"]
                actual_away = m["awayGoals"]

                if actual_home > actual_away:   actual_verdict = "L"
                elif actual_away > actual_home: actual_verdict = "V"
                else:                           actual_verdict = "E"

                probs = prediction["probabilidades"]
                if probs["local"] > probs["empate"] and probs["local"] > probs["visitante"]:
                    pred_verdict = "L"
                elif probs["visitante"] > probs["empate"] and probs["visitante"] > probs["local"]:
                    pred_verdict = "V"
                else:
                    pred_verdict = "E"

                winner_hit = (actual_verdict == pred_verdict)

                predicted_total = (
                    prediction["expected_goals"]["local"]
                    + prediction["expected_goals"]["visitante"]
                )
                actual_total = actual_home + actual_away
                goals_hit = abs(predicted_total - actual_total) <= 1.0

                return {
                    "match":        f"{m['homeTeam_name']} vs {m['awayTeam_name']}",
                    "date":         m["utcDate"],
                    "actual_score": f"{actual_home}-{actual_away}",
                    "prediction": {
                        "winner": pred_verdict,
                        "probs":  probs,
                        "xg":     prediction["expected_goals"],
                    },
                    "is_hit": winner_hit,
                    "details": {
                        "winner_hit": winner_hit,
                        "goals_hit":  goals_hit,
                    },
                    "source": "legacy",  # ⚠️  Data leakage present
                }
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                logger.error(f"[History] Legacy match processing error: {e}")
                return None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_match, recent_matches))

            history_results = [r for r in results if r is not None]
            hits   = sum(1 for r in history_results if r["is_hit"])
            misses = len(history_results) - hits

            return {
                "summary": {
                    "hits":     hits,
                    "misses":   misses,
                    "total":    len(history_results),
                    "accuracy": round((hits / len(history_results)) * 100, 1) if history_results else 0,
                    "source":   "legacy",
                    "note":     "⚠️ Precisión inflada: el modelo fue evaluado en datos de entrenamiento. Reentrena el modelo para obtener métricas reales.",
                },
                "history": history_results,
            }

        except HTTPException as e:
            if e.status_code == 202:
                return {
                    "training_in_progress": True,
                    "summary": {"hits": 0, "misses": 0, "total": 0, "accuracy": 0},
                    "history": [],
                }
            raise e


history_service = HistoryService()
