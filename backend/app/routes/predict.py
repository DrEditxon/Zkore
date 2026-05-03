from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends
import logging

from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service
from app.services.history_service import history_service
from app.services.picks_service import picks_service
from app.services.registry_service import registry_service
from app.services.performance_service import performance_service

from app.core.pipeline import predict_match
from app.core.limiter import limiter
from app.core.security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/predict")
@limiter.limit("30/minute")
def predict(
    request: Request,
    league_code: str,
    team_local: int,
    team_visitante: int,
    background_tasks: BackgroundTasks,
    match_id: int = None,
    utc_date: str = None,
    api_key: str = Depends(verify_api_key)
):
    if team_local == team_visitante:
        raise HTTPException(status_code=400, detail="Los equipos deben ser diferentes")

    standings = data_service.get_standings(league_code)
    home_name, away_name = "Local", "Visitante"
    if standings:
        try:
            for entry in standings["standings"][0]["table"]:
                if entry["team"]["id"] == team_local:    home_name = entry["team"]["name"]
                if entry["team"]["id"] == team_visitante: away_name = entry["team"]["name"]
        except KeyError:
            pass

    try:
        response_data = predict_match(
            league_code, team_local, team_visitante, home_name, away_name,
            background_tasks=background_tasks, match_id=match_id, utc_date=utc_date,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción completa: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del modelo de predicción")

    return response_data


@router.get("/history/{league_code}")
def get_history(league_code: str, api_key: str = Depends(verify_api_key)):
    return history_service.get_league_history(league_code)


@router.get("/value-bets/{league_code}")
def get_value_bets(league_code: str, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """
    Detects Value Bet opportunities by comparing our XGBoost model predictions
    against a baseline 'synthetic bookmaker'.
    
    Strategy:
    - Synthetic Bookmaker uses a basic Poisson model based on team-specific goal averages.
    - Bookmaker applies a 7% overround (increases implied probabilities by 7%).
    - Edge = Expected Value (EV) = (Model_Prob * Bookie_Odds) - 1
    - Value Bet = EV >= +5%
    """
    historical = data_service.get_historical_matches(league_code)
    if not historical:
        raise HTTPException(status_code=404, detail=f"No hay datos históricos para {league_code}")

    total = len(historical)
    league_home_goals = sum(m["homeGoals"] for m in historical) / max(1, total)
    league_away_goals = sum(m["awayGoals"] for m in historical) / max(1, total)

    # Calculate basic attack/defense strengths for all teams
    team_stats = {}
    for m in historical:
        h = m["homeTeam_id"]
        a = m["awayTeam_id"]
        if h not in team_stats: team_stats[h] = {"hg":0, "hc":0, "hm":0, "ag":0, "ac":0, "am":0}
        if a not in team_stats: team_stats[a] = {"hg":0, "hc":0, "hm":0, "ag":0, "ac":0, "am":0}
        team_stats[h]["hg"] += m["homeGoals"]
        team_stats[h]["hc"] += m["awayGoals"]
        team_stats[h]["hm"] += 1
        team_stats[a]["ag"] += m["awayGoals"]
        team_stats[a]["ac"] += m["homeGoals"]
        team_stats[a]["am"] += 1

    upcoming = data_service.get_predicted_upcoming(league_code, background_tasks)
    matches  = upcoming.get("matches", [])

    if upcoming.get("training_in_progress"):
        return {
            "training_in_progress": True,
            "training_message": upcoming.get("training_message", "Modelo entrenándose..."),
            "value_bets": [],
            "baseline": {},
        }

    MIN_EV = 0.05   # >= 5% Expected Value required
    OVERROUND = 1.07

    import scipy.stats
    import numpy as np

    value_bets = []
    for m in matches:
        pred = m.get("prediction")
        if not pred or m.get("training"):
            continue

        h_id = m["homeTeam"]["id"]
        a_id = m["awayTeam"]["id"]

        # BUG-04 FIX: League-average fallback for teams with no historical stats.
        # The old fallback {hg:1, hc:1, hm:1} produced:
        #   h_atk = (1/1) / league_home_goals ≈ 0.71  (below-average, arbitrary)
        # The new fallback uses actual league means, yielding attack/defense
        # strengths of exactly 1.0 — the correct neutral prior for unknown teams.
        h_fallback = {"hg": league_home_goals, "hc": league_away_goals, "hm": 1}
        a_fallback = {"ag": league_away_goals, "ac": league_home_goals, "am": 1}
        h_st = team_stats.get(h_id, h_fallback)
        a_st = team_stats.get(a_id, a_fallback)

        # Bookie simulation (Basic Poisson)
        h_atk = (h_st["hg"] / max(1, h_st["hm"])) / max(0.1, league_home_goals)
        a_def = (a_st["ac"] / max(1, a_st["am"])) / max(0.1, league_home_goals)
        a_atk = (a_st["ag"] / max(1, a_st["am"])) / max(0.1, league_away_goals)
        h_def = (h_st["hc"] / max(1, h_st["hm"])) / max(0.1, league_away_goals)

        exp_h = max(0.1, h_atk * a_def * league_home_goals)
        exp_a = max(0.1, a_atk * h_def * league_away_goals)

        prob_matrix = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                prob_matrix[i, j] = scipy.stats.poisson.pmf(i, exp_h) * scipy.stats.poisson.pmf(j, exp_a)

        base_home = np.sum(np.tril(prob_matrix, -1))
        base_draw = np.trace(prob_matrix)
        base_away = np.sum(np.triu(prob_matrix, 1))

        tot = base_home + base_draw + base_away
        if tot == 0: continue
        base_home /= tot
        base_draw /= tot
        base_away /= tot

        # True Overround Math: Bookie INCREASES implied prob to offer LOWER odds
        mkt_home = min(0.99, base_home * OVERROUND)
        mkt_draw = min(0.99, base_draw * OVERROUND)
        mkt_away = min(0.99, base_away * OVERROUND)

        odds_home = round(1 / mkt_home, 2)
        odds_draw = round(1 / mkt_draw, 2)
        odds_away = round(1 / mkt_away, 2)

        # Our XGBoost Model Probabilities
        p_home = pred["local"]     / 100
        p_draw = pred["empate"]    / 100
        p_away = pred["visitante"] / 100

        bets_for_match = []
        for outcome, model_p, market_p, mkt_odds, team_label in [
            ("Local Gana",     p_home, mkt_home, odds_home, m["homeTeam"]["name"]),
            ("Empate",         p_draw, mkt_draw, odds_draw, "Empate"),
            ("Visitante Gana", p_away, mkt_away, odds_away, m["awayTeam"]["name"]),
        ]:
            ev = (model_p * mkt_odds) - 1.0   # True Expected Value
            edge = model_p - market_p         # Probability edge

            if ev >= MIN_EV:
                # Kelly Criterion Formula: f* = (bp - q) / b
                # b = decimal odds - 1, p = probability of winning, q = probability of losing
                b_odds = mkt_odds - 1.0
                kelly_fraction = max(0.0, (b_odds * model_p - (1.0 - model_p)) / b_odds)
                
                # Fractional Kelly (1/4 Kelly) is the industry standard for reducing volatility
                safe_kelly = kelly_fraction * 0.25

                bets_for_match.append({
                    "outcome":        outcome,
                    "team_label":     team_label,
                    "model_prob":     round(model_p  * 100, 1),
                    "market_prob":    round(market_p * 100, 1),
                    "market_odds":    mkt_odds,
                    "edge":           round(edge * 100, 1),
                    "expected_value": round(ev   * 100, 1),
                    "kelly_stake":    round(safe_kelly * 100, 2),
                })

        if bets_for_match:
            bets_for_match.sort(key=lambda x: x["expected_value"], reverse=True)
            value_bets.append({
                "match_id":   m["id"],
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
                "home_crest": m["homeTeam"].get("crest", ""),
                "away_crest": m["awayTeam"].get("crest", ""),
                "utc_date":   m["utcDate"],
                "home_id":    m["homeTeam"]["id"],
                "away_id":    m["awayTeam"]["id"],
                "bets":       bets_for_match,
                "best_ev":    bets_for_match[0]["expected_value"],
            })

    value_bets.sort(key=lambda x: x["best_ev"], reverse=True)

    return {
        "value_bets": value_bets,
        "baseline": {
            "bookie_simulation": "Basic Poisson Model",
            "overround":      f"{int((OVERROUND - 1) * 100)}%",
            "total_matches":  total,
            "min_edge":       f"{int(MIN_EV * 100)}%",
        },
    }


# ── TOP 10 PICKS diarios ──────────────────────────────────────────────────────

@router.get("/top-picks")
@limiter.limit("20/minute")
def get_top_picks(
    request: Request,
    league_code: str | None = None,
    api_key: str = Depends(verify_api_key),
):
    """
    TOP 10 PICKS del día — ranking automático por scoring compuesto.

    score = (prob * 0.50) + (ev * 0.30) + (confidence * 0.20)

    Filtros: prob >= 52%, EV >= 4%.
    Opcional: ?league_code=PL para filtrar por liga.
    """
    try:
        picks = picks_service.get_cached_picks(league_code)
        return {
            "date":       __import__('datetime').date.today().isoformat(),
            "total":      len(picks),
            "league_filter": league_code,
            "picks":      picks,
            "scoring_formula": {
                "prob_weight":       0.50,
                "ev_weight":         0.30,
                "confidence_weight": 0.20,
                "min_prob":          0.52,
                "min_ev":            0.04,
            },
        }
    except Exception as e:
        logger.error(f"[/top-picks] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generando top picks")


# ── MODEL REGISTRY ────────────────────────────────────────────────────────────

@router.get("/models")
def get_active_models(api_key: str = Depends(verify_api_key)):
    """
    Devuelve los modelos activos — versión, métricas y timestamp de entrenamiento.
    Sirve tanto desde Supabase (registry) como desde caché local como fallback.
    """
    try:
        models = registry_service.get_active_models(limit=30)

        # Enrich with live cache status
        live_status = {}
        for lc, payload in model_service._model_cache.items():
            live_status[lc] = {
                "cached_in_memory": True,
                "age_days":         payload.get("model_age_days"),
                "is_stale":         payload.get("is_stale"),
                "training_in_progress": model_service._training_in_progress.get(lc, False),
            }

        return {
            "version":      model_service.MODEL_VERSION,
            "registry":     models,
            "live_status":  live_status,
        }
    except Exception as e:
        logger.error(f"[/models] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error obteniendo modelos")


# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────

@router.get("/performance")
def get_performance(
    days: int = 30,
    api_key: str = Depends(verify_api_key),
):
    """
    Métricas globales de rendimiento: accuracy, ROI, Brier Score.
    ?days=30 (default) para el período de análisis.
    """
    try:
        result = performance_service.get_global_performance(days=days)
        result["period_days"] = days
        return result
    except Exception as e:
        logger.error(f"[/performance] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error calculando métricas")


@router.get("/performance/{league_code}")
def get_league_performance(
    league_code: str,
    days: int = 30,
    api_key: str = Depends(verify_api_key),
):
    """
    Métricas de rendimiento por liga específica.
    """
    try:
        result = performance_service.get_league_performance(league_code, days=days)
        result["period_days"] = days
        result["league_code"] = league_code
        return result
    except Exception as e:
        logger.error(f"[/performance/{league_code}] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error calculando métricas de liga")
