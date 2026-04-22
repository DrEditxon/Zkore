import logging
from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service
from app.services.feature_service import feature_service

logger = logging.getLogger(__name__)

def generate_heuristic_explanation(lambda_h, lambda_a, h_name, a_name, p_h, p_a):
    if p_h > 0.6:
        return f"{h_name} proyecta un amplio dominio debido a mayores expectativas de gol sostenidas localmente ({lambda_h:.2f} xG)."
    elif p_a > 0.6:
        return f"{a_name} tiene supremacía estadística y táctica en este emparejamiento con {lambda_a:.2f} xG."
    elif abs(p_h - p_a) < 0.1 and lambda_h > 1.5 and lambda_a > 1.5:
        return f"Encuentro altamente ofensivo y nivelado. Choque de dos fuertes ataques."
    elif abs(p_h - p_a) < 0.1:
        return f"Duelo cerrado. Las probabilidades muestran alta paridad táctica."
    else:
        fav = h_name if p_h > p_a else a_name
        return f"Ligero favoritismo para {fav} sustentado en métricas históricas de ELO defensivo/ofensivo cruzado."

def predict_match(league_code: str, home_id: int, away_id: int, home_name: str, away_name: str, apply_dixon_coles=False, background_tasks=None) -> dict:
    """
    Facade que orquesta TODO el ciclo predictivo de MLS (Machine Learning Service).
    Data -> Feature_Engineering -> Regressor -> Poisson -> Calibrator -> Explicación.
    Previene inconsistencias y duplicación de código en routing.
    """
    logger.info(f"[{league_code}] Validating and predicting match {home_id} vs {away_id}")
    
    # 1. Prediction execution
    lambda_h, lambda_a, payload = model_service.predict_xg(league_code, home_id, away_id, background_tasks)
    n_rows = payload.get("n_rows", 0)
    
    # 2. Probability Generation
    # Usar Dixon-Coles desactivado por defecto para estabilidad, a menos que se fuerce.
    rho_val = 0.0 # Standard Poisson
    prob_matrix = poisson_service.calculate_probability_matrix(lambda_h, lambda_a, rho=rho_val)
    markets = poisson_service.extract_metrics(prob_matrix)
    top_scores = poisson_service.get_top_scorelines(prob_matrix)
    goal_distributions = poisson_service.format_goal_distributions(lambda_h, lambda_a)
    
    p_awy = markets["prob_away_win"]
    p_drw = markets["prob_draw"]
    p_hom = markets["prob_home_win"]
    
    # 3. Probability Calibration (If Available)
    calibrator = payload.get("calibrator")
    if calibrator:
        raw_probs = [[p_awy/100, p_drw/100, p_hom/100]]
        calib = calibrator.predict_proba(raw_probs)[0]
        p_awy = round(calib[0] * 100, 2)
        p_drw = round(calib[1] * 100, 2)
        p_hom = round(calib[2] * 100, 2)
        # Update markets to match calibrated
        markets["prob_away_win"] = p_awy
        markets["prob_draw"] = p_drw
        markets["prob_home_win"] = p_hom

    # 4. Explicabilidad Simple Heurística
    explicacion = generate_heuristic_explanation(lambda_h, lambda_a, home_name, away_name, p_hom/100, p_awy/100)

    # 5. External RapidAPI Fetch
    rapid_data = data_service.get_expected_match_stats(home_name, away_name)

    conf = "Alta" if n_rows >= 100 else ("Media" if n_rows >= 50 else "Baja")

    result = {
        "expected_goals": {
            "local": round(lambda_h, 2),
            "visitante": round(lambda_a, 2)
        },
        "probabilidades": {
            "local": p_hom,
            "empate": p_drw,
            "visitante": p_awy,
        },
        "metricas_mercado": markets,
        "distribucion_goles": goal_distributions,
        "marcadores_probables": top_scores,
        "modelo_info": {
            "partidos_entrenados": n_rows,
            "confianza": conf,
            "tipo": "XGBoost Regressor + Poisson" + (" + Platt Calibrator" if calibrator else ""),
            "explicacion": explicacion
        }
    }
    
    # Integrar RapidAPI al final
    result.update(rapid_data)
    
    return result
