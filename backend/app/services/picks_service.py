"""
Picks Service — Generación de TOP 10 PICKS diarios.

Scoring compuesto multi-factor:

    score_final = (prob * 0.50) + (ev * 0.30) + (confidence * 0.20)

Donde:
  - prob        = probabilidad del resultado predicho (normalizada a [0,1])
  - ev          = Expected Value contra cuota bookie simulada ([0, ~0.35])
  - confidence  = nivel de confianza del modelo (0.3 / 0.6 / 1.0)

Filtros mínimos:
  - prob >= 0.52  (al menos 52% de probabilidad)
  - ev   >= 0.04  (al menos 4% de valor esperado)

Output: lista de hasta 10 picks ordenados por score_final desc,
        enriquecidos con todos los datos de UI relevantes.

Persistencia: guarda en Supabase tabla `top_picks` (upsert diario).
"""

import logging
import scipy.stats
import numpy as np
from datetime import datetime, timezone, date

logger = logging.getLogger(__name__)

# ── Scoring weights ───────────────────────────────────────────────────────────
W_PROB       = 0.50
W_EV         = 0.30
W_CONFIDENCE = 0.20

# ── Filtros ────────────────────────────────────────────────────────────────────
MIN_PROB = 0.52   # 52% mínimo
MIN_EV   = 0.04   # 4% EV mínimo

# ── Overround simulado del bookie ─────────────────────────────────────────────
OVERROUND = 1.07

# ── Confidence score mapping ─────────────────────────────────────────────────
CONF_SCORES = {"Alta": 1.0, "Media": 0.6, "Baja": 0.3}

TOP_N = 10


def _confidence_from_mae(mae_home: float | None, mae_away: float | None, n_rows: int = 0) -> str:
    """Mapea MAE del modelo a nivel de confianza textual."""
    if mae_home is None or mae_away is None:
        return "Baja"
    avg = (mae_home + mae_away) / 2
    if avg < 0.8 and n_rows >= 100:
        return "Alta"
    elif avg < 1.1 and n_rows >= 50:
        return "Media"
    return "Baja"


def _poisson_probs(exp_h: float, exp_a: float) -> tuple[float, float, float]:
    """Calcula probabilidades 1X2 via Poisson para la simulación bookie."""
    prob_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            prob_matrix[i, j] = scipy.stats.poisson.pmf(i, exp_h) * scipy.stats.poisson.pmf(j, exp_a)
    base_home = float(np.sum(np.tril(prob_matrix, -1)))
    base_draw = float(np.trace(prob_matrix))
    base_away = float(np.sum(np.triu(prob_matrix, 1)))
    tot = base_home + base_draw + base_away or 1
    return base_home / tot, base_draw / tot, base_away / tot


class PicksService:
    """
    Genera el ranking diario TOP 10 de picks usando el scoring compuesto.
    Consume las predicciones calculadas por data_service y aplica el motor
    de scoring sin necesidad de re-ejecutar el modelo.
    """

    def generate_top_picks(
        self,
        league_code: str | None = None,
        save_to_db: bool = True,
    ) -> list[dict]:
        """
        Genera los TOP 10 picks del día.

        Args:
            league_code: si se especifica, solo para esa liga.
                         Si es None, para todas las ligas configuradas.
            save_to_db:  si True, guarda en Supabase tabla `top_picks`.

        Returns:
            Lista de picks ordenados por score_final desc (max 10).
        """
        from app.services.data_service import data_service
        from app.services.model_service import model_service
        from app.core.config import settings

        leagues = [league_code] if league_code else list(settings.LEAGUES_METADATA.keys())
        candidates = []

        for lc in leagues:
            try:
                upcoming = data_service.get_predicted_upcoming(lc)
            except Exception as e:
                logger.warning(f"[Picks] [{lc}] Error getting upcoming: {e}")
                continue

            if upcoming.get("training_in_progress"):
                continue

            # Get model metadata for confidence scoring
            payload = model_service._model_cache.get(lc)
            mae_home = payload.get("mae_home") if payload else None
            mae_away = payload.get("mae_away") if payload else None
            n_rows   = payload.get("n_rows", 0) if payload else 0

            # Get historical data for bookie simulation
            try:
                historical = data_service.get_historical_matches(lc)
                total = max(1, len(historical))
                lg_home = sum(m["homeGoals"] for m in historical) / total
                lg_away = sum(m["awayGoals"] for m in historical) / total

                team_stats: dict = {}
                for m in historical:
                    h, a = m["homeTeam_id"], m["awayTeam_id"]
                    if h not in team_stats:
                        team_stats[h] = {"hg": 0, "hc": 0, "hm": 0}
                    if a not in team_stats:
                        team_stats[a] = {"ag": 0, "ac": 0, "am": 0}
                    team_stats[h]["hg"] += m["homeGoals"]
                    team_stats[h]["hc"] += m["awayGoals"]
                    team_stats[h]["hm"] += 1
                    team_stats[a].setdefault("ag", 0)
                    team_stats[a].setdefault("ac", 0)
                    team_stats[a].setdefault("am", 0)
                    team_stats[a]["ag"] += m["awayGoals"]
                    team_stats[a]["ac"] += m["homeGoals"]
                    team_stats[a]["am"] += 1
            except Exception:
                lg_home, lg_away, team_stats = 1.3, 1.1, {}

            for m in upcoming.get("matches", []):
                pred = m.get("prediction")
                if not pred or m.get("training"):
                    continue

                p_home = pred.get("local", 33.3) / 100
                p_draw = pred.get("empate", 33.3) / 100
                p_away = pred.get("visitante", 33.3) / 100

                # Which outcome does the model favour?
                if p_home >= p_draw and p_home >= p_away:
                    verdict, model_p, outcome_label = "L", p_home, "Local Gana"
                elif p_away > p_home and p_away > p_draw:
                    verdict, model_p, outcome_label = "V", p_away, "Visitante Gana"
                else:
                    verdict, model_p, outcome_label = "E", p_draw, "Empate"

                if model_p < MIN_PROB:
                    continue

                # Bookie simulation via Poisson
                h_id = m["homeTeam"]["id"]
                a_id = m["awayTeam"]["id"]
                h_st = team_stats.get(h_id, {"hg": lg_home, "hc": lg_away, "hm": 1})
                a_st = team_stats.get(a_id, {"ag": lg_away, "ac": lg_home, "am": 1})

                h_atk = (h_st.get("hg", lg_home) / max(1, h_st.get("hm", 1))) / max(0.1, lg_home)
                a_def = (a_st.get("ac", lg_home) / max(1, a_st.get("am", 1))) / max(0.1, lg_home)
                a_atk = (a_st.get("ag", lg_away) / max(1, a_st.get("am", 1))) / max(0.1, lg_away)
                h_def = (h_st.get("hc", lg_away) / max(1, h_st.get("hm", 1))) / max(0.1, lg_away)

                exp_h = max(0.1, h_atk * a_def * lg_home)
                exp_a = max(0.1, a_atk * h_def * lg_away)

                bk_home, bk_draw, bk_away = _poisson_probs(exp_h, exp_a)

                # Apply overround
                mkt_h = min(0.99, bk_home * OVERROUND)
                mkt_d = min(0.99, bk_draw * OVERROUND)
                mkt_a = min(0.99, bk_away * OVERROUND)

                if verdict == "L":
                    mkt_prob = mkt_h
                    mkt_odds = round(1 / mkt_h, 2)
                elif verdict == "V":
                    mkt_prob = mkt_a
                    mkt_odds = round(1 / mkt_a, 2)
                else:
                    mkt_prob = mkt_d
                    mkt_odds = round(1 / mkt_d, 2)

                ev   = (model_p * mkt_odds) - 1.0
                edge = model_p - mkt_prob

                if ev < MIN_EV:
                    continue

                confidence = _confidence_from_mae(mae_home, mae_away, n_rows)
                conf_score = CONF_SCORES[confidence]

                # Normalize EV to [0, 1] range (cap at 35% EV for normalization)
                ev_norm = min(ev / 0.35, 1.0)

                score_final = (
                    model_p   * W_PROB +
                    ev_norm   * W_EV +
                    conf_score * W_CONFIDENCE
                )

                # Kelly stake (fractional 1/4)
                b_odds = mkt_odds - 1.0
                kelly = max(0.0, (b_odds * model_p - (1.0 - model_p)) / b_odds) * 0.25

                candidates.append({
                    "match_id":      m["id"],
                    "league_code":   lc,
                    "home_team":     m["homeTeam"]["name"],
                    "away_team":     m["awayTeam"]["name"],
                    "home_crest":    m["homeTeam"].get("crest", ""),
                    "away_crest":    m["awayTeam"].get("crest", ""),
                    "utc_date":      m["utcDate"],
                    "verdict":       verdict,
                    "outcome_label": outcome_label,
                    "prob_home":     round(p_home * 100, 1),
                    "prob_draw":     round(p_draw * 100, 1),
                    "prob_away":     round(p_away * 100, 1),
                    "model_prob":    round(model_p * 100, 1),
                    "market_prob":   round(mkt_prob * 100, 1),
                    "market_odds":   mkt_odds,
                    "edge":          round(edge * 100, 1),
                    "expected_value": round(ev * 100, 1),
                    "kelly_stake":   round(kelly * 100, 2),
                    "confidence":    confidence,
                    "score_final":   round(score_final, 4),
                    "xg_local":      pred.get("_xg_local"),
                    "xg_visitante":  pred.get("_xg_visitante"),
                    "is_live":       m.get("is_live", False),
                    "live_score":    m.get("live_score"),
                    "live_minute":   m.get("live_minute"),
                })

        # Sort and take top N
        candidates.sort(key=lambda x: x["score_final"], reverse=True)
        top_picks = candidates[:TOP_N]

        # Add rank
        for idx, pick in enumerate(top_picks, 1):
            pick["rank"] = idx

        # Persist to Supabase (fire-and-forget)
        if save_to_db and top_picks:
            try:
                self._save_top_picks(top_picks)
            except Exception as e:
                logger.warning(f"[Picks] Failed to save to DB (non-fatal): {e}")

        logger.info(
            f"[Picks] Generated {len(top_picks)} top picks "
            f"from {len(candidates)} candidates across {len(leagues)} leagues."
        )
        return top_picks

    def _save_top_picks(self, picks: list[dict]) -> None:
        """Upserta los picks del día en la tabla `top_picks` de Supabase."""
        from app.services.supabase_service import supabase_service
        if not supabase_service.enabled:
            return

        today_str = date.today().isoformat()
        rows = [
            {
                "date":           today_str,
                "match_id":       p["match_id"],
                "league_code":    p["league_code"],
                "home_team":      p["home_team"],
                "away_team":      p["away_team"],
                "home_crest":     p.get("home_crest", ""),
                "away_crest":     p.get("away_crest", ""),
                "utc_date":       p["utc_date"],
                "verdict":        p["verdict"],
                "score":          p["score_final"],
                "rank":           p["rank"],
                "prob_home":      p["prob_home"],
                "prob_draw":      p["prob_draw"],
                "prob_away":      p["prob_away"],
                "model_prob":     p["model_prob"],
                "market_odds":    p["market_odds"],
                "kelly_stake":    p["kelly_stake"],
                "outcome_label":  p["outcome_label"],
                "expected_value": p["expected_value"],
                "confidence":     p["confidence"],
            }
            for p in picks
        ]

        try:
            endpoint = f"{supabase_service.url}/rest/v1/top_picks"
            r = supabase_service.session.post(
                endpoint,
                headers={
                    **supabase_service.headers,
                    "Prefer": "resolution=merge-duplicates",
                },
                json=rows,
                timeout=10,
            )
            if r.status_code in (200, 201, 204):
                logger.info(f"[Picks] Saved {len(rows)} picks for {today_str}")
            else:
                logger.warning(f"[Picks] DB save failed: {r.status_code} {r.text[:150]}")
        except Exception as e:
            logger.error(f"[Picks] _save_top_picks error: {e}")

    def get_cached_picks(self, league_code: str | None = None) -> list[dict]:
        """
        Intenta obtener los picks del día desde Supabase.
        Si no hay picks recientes o Supabase no está disponible,
        los genera en tiempo real.
        """
        from app.services.supabase_service import supabase_service

        today_str = date.today().isoformat()

        if supabase_service.enabled:
            try:
                params = f"date=eq.{today_str}&order=rank.asc&limit=10"
                if league_code:
                    params += f"&league_code=eq.{league_code}"
                endpoint = f"{supabase_service.url}/rest/v1/top_picks?{params}"
                r = supabase_service.session.get(endpoint, headers=supabase_service.headers, timeout=8)
                if r.status_code == 200:
                    cached = r.json()
                    if cached:
                        logger.info(f"[Picks] Serving {len(cached)} cached picks from Supabase.")
                        return cached
            except Exception as e:
                logger.warning(f"[Picks] Cache fetch failed: {e}")

        # Generate fresh
        return self.generate_top_picks(league_code, save_to_db=True)


picks_service = PicksService()
