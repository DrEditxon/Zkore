import logging
from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service
from app.services.feature_service import feature_service
from app.services.supabase_service import supabase_service

logger = logging.getLogger(__name__)


def generate_heuristic_explanation(lambda_h, lambda_a, h_name, a_name, p_h, p_a):
    if p_h > 0.6:
        return f"{h_name} proyecta un amplio dominio debido a mayores expectativas de gol sostenidas localmente ({lambda_h:.2f} xG)."
    elif p_a > 0.6:
        return f"{a_name} tiene supremacía estadística y táctica en este emparejamiento con {lambda_a:.2f} xG."
    elif abs(p_h - p_a) < 0.1 and lambda_h > 1.5 and lambda_a > 1.5:
        return "Encuentro altamente ofensivo y nivelado. Choque de dos fuertes ataques."
    elif abs(p_h - p_a) < 0.1:
        return "Duelo cerrado. Las probabilidades muestran alta paridad táctica."
    else:
        fav = h_name if p_h > p_a else a_name
        return f"Ligero favoritismo para {fav} sustentado en métricas históricas de ELO defensivo/ofensivo cruzado."


def predict_match(
    league_code: str,
    home_id: int,
    away_id: int,
    home_name: str,
    away_name: str,
    background_tasks=None,
    include_rapid_stats: bool = True,
    match_id: int = None,
    utc_date: str = None,
) -> dict:
    """
    Full prediction pipeline: Data → Features → XGBoost → Poisson → Calibrator.

    `include_rapid_stats=False` skips RapidAPI calls for the upcoming grid.
    The modal uses True and fetches stats on demand.
    """
    logger.info(f"[{league_code}] Predicting {home_id} vs {away_id}")

    lambda_h, lambda_a, payload = model_service.predict_xg(
        league_code, home_id, away_id, background_tasks
    )
    n_rows = payload.get("n_rows", 0)

    prob_matrix = poisson_service.calculate_probability_matrix(lambda_h, lambda_a, rho=0.0)
    markets     = poisson_service.extract_metrics(prob_matrix)
    top_scores  = poisson_service.get_top_scorelines(prob_matrix)
    goal_dists  = poisson_service.format_goal_distributions(lambda_h, lambda_a)

    p_awy = markets["prob_away_win"]
    p_drw = markets["prob_draw"]
    p_hom = markets["prob_home_win"]

    calibrator = payload.get("calibrator")
    if calibrator:
        raw_probs = [[p_awy / 100, p_drw / 100, p_hom / 100]]
        calib = calibrator.predict_proba(raw_probs)[0]
        p_awy = round(calib[0] * 100, 2)
        p_drw = round(calib[1] * 100, 2)
        p_hom = round(calib[2] * 100, 2)
        markets["prob_away_win"] = p_awy
        markets["prob_draw"]     = p_drw
        markets["prob_home_win"] = p_hom

    explicacion = generate_heuristic_explanation(
        lambda_h, lambda_a, home_name, away_name, p_hom / 100, p_awy / 100
    )

    mae_home = payload.get("mae_home", 999)
    mae_away = payload.get("mae_away", 999)
    mae_avg  = (mae_home + mae_away) / 2

    if mae_home < 0.8 and mae_away < 0.8 and n_rows >= 100:
        conf = "Alta"
    elif mae_avg < 1.1 and n_rows >= 50:
        conf = "Media"
    else:
        conf = "Baja"

    result = {
        "expected_goals": {
            "local":    round(lambda_h, 2),
            "visitante": round(lambda_a, 2),
        },
        "probabilidades": {
            "local":     p_hom,
            "empate":    p_drw,
            "visitante": p_awy,
        },
        "metricas_mercado":    markets,
        "distribucion_goles":  goal_dists,
        "marcadores_probables": top_scores,
        "modelo_info": {
            "partidos_entrenados": n_rows,
            "confianza":  conf,
            "tipo":       "XGBoost Regressor + Poisson" + (" + Platt Calibrator" if calibrator else ""),
            "explicacion": explicacion,
            "mae_home":   round(mae_home, 3) if mae_home != 999 else None,
            "mae_away":   round(mae_away, 3) if mae_away != 999 else None,
            "trained_at": payload.get("trained_at"),
            "model_age_days": payload.get("model_age_days"),
        },
    }

    # PERF FIX E: Only hit RapidAPI when the user explicitly opens the match modal.
    # Skip it entirely for the bulk upcoming-grid load.
    if include_rapid_stats:
        rapid_data = data_service.get_expected_match_stats(home_name, away_name, league_code)
        result.update(rapid_data)

    # ─── Supabase Persistence ────────────────────────────────────────────────
    if background_tasks and match_id:
        p_hom = result["probabilidades"]["local"]
        p_drw = result["probabilidades"]["empate"]
        p_awy = result["probabilidades"]["visitante"]
        
        if p_hom > p_drw and p_hom > p_awy:     verdict = "L"
        elif p_awy > p_hom and p_awy > p_drw:   verdict = "V"
        else:                                     verdict = "E"

        db_payload = {
            "match_id": match_id,
            "league_code": league_code,
            "home_team": home_name,
            "away_team": away_name,
            "prediction": result["probabilidades"],
            "prob_home": p_hom,
            "prob_draw": p_drw,
            "prob_away": p_awy,
            "verdict": verdict,
            "utc_date": utc_date
        }
        background_tasks.add_task(supabase_service.save_prediction, db_payload)

    return result
