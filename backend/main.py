import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .model import predict_match
from .data_fetcher import get_leagues, get_standings, get_upcoming_matches
from .rapidapi_fetcher import get_expected_match_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.get("/leagues")
def list_leagues():
    return get_leagues()

@app.get("/upcoming/{league_code}")
def list_upcoming(league_code: str):
    data = get_upcoming_matches(league_code)
    if not data["matches"]:
        return data # No upcoming matches

    # Add quick prediction to each match
    for m in data["matches"]:
        try:
            # We call predict_match but we only care about outcome probabilities
            p = predict_match(league_code, m["homeTeam"]["id"], m["awayTeam"]["id"])
            m["prediction"] = p["probabilidades"]
            # Summarize result type for quick view
            probs = p["probabilidades"]
            if probs["local"] > probs["empate"] and probs["local"] > probs["visitante"]:
                m["verdict"] = "L"
            elif probs["visitante"] > probs["local"] and probs["visitante"] > probs["empate"]:
                m["verdict"] = "V"
            else:
                m["verdict"] = "E"
        except Exception:
            m["prediction"] = {"local": 33, "empate": 33, "visitante": 33}
            m["verdict"] = "?"

    return data

@app.get("/predict")
def predict(league_code: str, team_local: int, team_visitante: int):
    if team_local == team_visitante:
        raise HTTPException(status_code=400, detail="Los equipos deben ser diferentes")

    try:
        result = predict_match(league_code, team_local, team_visitante)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del modelo de predicción")

    # Get Team Names for RapidAPI
    standings = get_standings(league_code)
    home_name, away_name = "Local", "Visitante"
    if standings:
        for entry in standings['standings'][0]['table']:
            if entry['team']['id'] == team_local: home_name = entry['team']['name']
            if entry['team']['id'] == team_visitante: away_name = entry['team']['name']

    # Fetch API-Football data individual metrics
    try:
        rapid_data = get_expected_match_stats(home_name, away_name)
    except Exception as e:
        logger.error(f"RapidAPI fetcher failed completely: {e}")
        rapid_data = {}

    response_data = {
        "probabilidades":       result["probabilidades"],
        "metricas_mercado":     result["metricas_mercado"],
        "distribucion_goles":   result["distribucion_goles"],
        "marcadores_probables": result["marcadores_probables"],
        "modelo_info":          result["modelo_info"],
    }
    response_data.update(rapid_data)
    
    return response_data