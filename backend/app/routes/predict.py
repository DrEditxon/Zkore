from fastapi import APIRouter, HTTPException
import logging

from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service

from app.core.pipeline import predict_match

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/predict")
def predict(league_code: str, team_local: int, team_visitante: int):
    # Parameter names match the user's requested specification
    if team_local == team_visitante:
        raise HTTPException(status_code=400, detail="Los equipos deben ser diferentes")

    # Name Resolution Helper
    standings = data_service.get_standings(league_code)
    home_name, away_name = "Local", "Visitante"
    if standings:
        try:
            for entry in standings['standings'][0]['table']:
                if entry['team']['id'] == team_local: home_name = entry['team']['name']
                if entry['team']['id'] == team_visitante: away_name = entry['team']['name']
        except KeyError:
            pass

    try:
        response_data = predict_match(league_code, team_local, team_visitante, home_name, away_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción completa: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno robusto del modelo de predicción")

    return response_data
