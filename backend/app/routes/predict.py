from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
import logging

from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service
from app.services.history_service import history_service

from app.core.pipeline import predict_match
from app.core.limiter import limiter

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
def get_history(league_code: str):
    return history_service.get_league_history(league_code)


@router.get("/value-bets/{league_code}")
def get_value_bets(league_code: str, background_tasks: BackgroundTasks):
    """
    Detects Value Bet opportunities by comparing XGBoost model predictions
    against a baseline 'market' built from historical league win/draw/loss rates.

    Strategy:
    - Market implied prob = historical rate ÷ overround (7%)
    - Market odds        = 1 / market_implied_prob
    - Edge               = model_probability − market_implied_probability
    - Value Bet          = edge ≥ 7%  AND  expected_value > 0

    This works because the XGBoost model uses team-specific features
    (ELO, form, H2H) while the baseline is the raw league average.
    A significant deviation signals a genuine edge vs the "market".
    """
    # ── Historical baseline ───────────────────────────────────────────────
    historical = data_service.get_historical_matches(league_code)
    if not historical:
        raise HTTPException(status_code=404, detail=f"No hay datos históricos para {league_code}")

    total     = len(historical)
    home_wins = sum(1 for m in historical if m["homeGoals"] > m["awayGoals"])
    draws     = sum(1 for m in historical if m["homeGoals"] == m["awayGoals"])
    away_wins = sum(1 for m in historical if m["homeGoals"] < m["awayGoals"])

    base_home = home_wins / total
    base_draw = draws     / total
    base_away = away_wins / total

    # Simulate bookmaker overround (7%)
    OVERROUND = 1.07
    mkt_home = base_home / OVERROUND
    mkt_draw = base_draw / OVERROUND
    mkt_away = base_away / OVERROUND

    odds_home = round(1 / mkt_home, 2)
    odds_draw = round(1 / mkt_draw, 2)
    odds_away = round(1 / mkt_away, 2)

    # ── Upcoming matches with predictions ─────────────────────────────────
    upcoming = data_service.get_predicted_upcoming(league_code, background_tasks)
    matches  = upcoming.get("matches", [])

    if upcoming.get("training_in_progress"):
        return {
            "training_in_progress": True,
            "training_message": upcoming.get("training_message", "Modelo entrenándose..."),
            "value_bets": [],
            "baseline": {},
        }

    MIN_EDGE = 0.07   # ≥ 7% edge over the market required

    value_bets = []
    for m in matches:
        pred = m.get("prediction")
        if not pred or m.get("training"):
            continue

        p_home = pred["local"]     / 100
        p_draw = pred["empate"]    / 100
        p_away = pred["visitante"] / 100

        bets_for_match = []
        for outcome, model_p, market_p, mkt_odds, team_label in [
            ("Local Gana",     p_home, mkt_home, odds_home, m["homeTeam"]["name"]),
            ("Empate",         p_draw, mkt_draw, odds_draw, "Empate"),
            ("Visitante Gana", p_away, mkt_away, odds_away, m["awayTeam"]["name"]),
        ]:
            edge = model_p - market_p
            ev   = model_p * mkt_odds - 1   # Expected Value

            if edge >= MIN_EDGE and ev > 0:
                bets_for_match.append({
                    "outcome":        outcome,
                    "team_label":     team_label,
                    "model_prob":     round(model_p  * 100, 1),
                    "market_prob":    round(market_p * 100, 1),
                    "market_odds":    mkt_odds,
                    "edge":           round(edge * 100, 1),
                    "expected_value": round(ev   * 100, 1),
                })

        if bets_for_match:
            bets_for_match.sort(key=lambda x: x["edge"], reverse=True)
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
                "best_edge":  bets_for_match[0]["edge"],
            })

    value_bets.sort(key=lambda x: x["best_edge"], reverse=True)

    return {
        "value_bets": value_bets,
        "baseline": {
            "home_win_rate":  round(base_home * 100, 1),
            "draw_rate":      round(base_draw * 100, 1),
            "away_win_rate":  round(base_away * 100, 1),
            "odds_home":      odds_home,
            "odds_draw":      odds_draw,
            "odds_away":      odds_away,
            "overround":      f"{int((OVERROUND - 1) * 100)}%",
            "total_matches":  total,
            "min_edge":       f"{int(MIN_EDGE * 100)}%",
        },
    }
