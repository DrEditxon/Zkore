import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import hashlib
import os
import json
from app.core.config import settings

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class DataService:
    def __init__(self):
        self.headers_football_data = {"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY}
        self.headers_rapidapi = {
            "x-rapidapi-key": settings.RAPIDAPI_KEY,
            "x-rapidapi-host": settings.RAPIDAPI_HOST
        }
        
        # Robust HTTP session with retries
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504, 429])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get_cache_path(self, key):
        return os.path.join(CACHE_DIR, f"{key}.json")

    def _get_from_cache(self, key, ttl=None):
        if ttl is None:
            ttl = settings.CACHE_DURATION
            
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] < ttl:
                    return entry["data"]
            except Exception as e:
                logger.warning(f"Error reading cache for {key}: {e}")
        return None

    def _set_to_cache(self, key, data):
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"data": data, "timestamp": time.time()}, f)
        except Exception as e:
            logger.warning(f"Error writing cache for {key}: {e}")

    def get_leagues(self):
        return [{"code": code, "name": meta["name"], "flag": meta["flag"]} for code, meta in settings.LEAGUES_METADATA.items()]

    def get_upcoming_matches(self, league_code: str, limit: int = 10) -> dict:
        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        params = {"status": "SCHEDULED"}
        try:
            response = self.session.get(url, headers=self.headers_football_data, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching upcoming matches for {league_code}: {e}")
            return {"matchday": 0, "matches": []}

        data = response.json()
        matches_raw = data.get("matches", [])
        if not matches_raw:
            return {"matchday": 0, "matches": []}

        current_matchday = matches_raw[0].get("matchday", 0)
        
        matches = []
        for m in matches_raw[:limit]:
            matches.append({
                "id": m["id"],
                "utcDate": m["utcDate"],
                "homeTeam": {
                    "id": m["homeTeam"]["id"],
                    "name": m["homeTeam"]["name"],
                    "crest": m["homeTeam"]["crest"]
                },
                "awayTeam": {
                    "id": m["awayTeam"]["id"],
                    "name": m["awayTeam"]["name"],
                    "crest": m["awayTeam"]["crest"]
                }
            })
        
        return {"matchday": current_matchday, "matches": matches}

    def get_standings(self, league_code: str):
        cached_data = self._get_from_cache(f"standings_{league_code}")
        if cached_data:
            return cached_data

        url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
        try:
            response = self.session.get(url, headers=self.headers_football_data, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching standings for {league_code}: {e}")
            return None
        
        data = response.json()
        self._set_to_cache(f"standings_{league_code}", data)
        return data

    def get_historical_matches(self, league_code: str) -> list:
        cached = self._get_from_cache(f"matches_{league_code}", ttl=settings.CACHE_DURATION_HISTORICAL)
        if cached:
            return cached

        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        params = {"status": "FINISHED"}
        try:
            response = self.session.get(url, headers=self.headers_football_data, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical matches for {league_code}: {e}")
            return []

        data = response.json()
        matches = []
        for m in data.get("matches", []):
            score = m.get("score", {})
            full = score.get("fullTime", {})
            home_goals = full.get("home")
            away_goals = full.get("away")
            if home_goals is None or away_goals is None:
                continue
            matches.append({
                "utcDate": m["utcDate"],
                "homeTeam_id": m["homeTeam"]["id"],
                "homeTeam_name": m["homeTeam"]["name"],
                "awayTeam_id": m["awayTeam"]["id"],
                "awayTeam_name": m["awayTeam"]["name"],
                "homeGoals": home_goals,
                "awayGoals": away_goals,
            })

        self._set_to_cache(f"matches_{league_code}", matches)
        return matches

    def get_expected_match_stats(self, home_name: str, away_name: str):
        rate_limit = "Desconocido"
        try:
            status_req = self.session.get(f"https://{settings.RAPIDAPI_HOST}/v3/timezone", headers=self.headers_rapidapi, timeout=4)
            if "x-ratelimit-requests-remaining" in status_req.headers:
                rate_limit = status_req.headers["x-ratelimit-requests-remaining"]
                
            if status_req.status_code == 403:
                rate_limit = "403 (Suscripción Pendiente)"
            elif status_req.status_code == 429:
                rate_limit = "Límite Excedido"
                
        except Exception:
            rate_limit = "Error de Conexión"

        return {
            "estadisticas_esperadas": None,
            "nota": "Estadísticas en tiempo real próximamente disponibles",
            "rapidapi_rate_limit": rate_limit
        }

    def invalidate_cache(self, league_code: str):
        for prefix in ["matches", "standings"]:
            key = f"{prefix}_{league_code}"
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.warning(f"Error invalidating cache for {key}: {e}")

data_service = DataService()
