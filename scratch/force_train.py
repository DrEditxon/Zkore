import os
import sys
import logging

# Forzar buffer a stdout
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from app.core.config import settings
from app.services.data_service import data_service
from app.services.model_service import model_service

def force_retrain():
    print("Iniciando re-entrenamiento forzado de todas las ligas...", flush=True)
    for league in settings.LEAGUES_METADATA.keys():
        print(f"\n--- Entrenando {league} ---", flush=True)
        try:
            matches = data_service.get_historical_matches(league)
            if not matches:
                print(f"No hay datos históricos para {league}", flush=True)
                continue
            model_service._train_and_save(league, matches)
            print(f"✓ {league} entrenado exitosamente con nuevo objetivo Poisson.", flush=True)
        except Exception as e:
            print(f"Error entrenando {league}: {e}", flush=True)

if __name__ == "__main__":
    force_retrain()
