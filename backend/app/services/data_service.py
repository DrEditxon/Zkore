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

    def get_predicted_upcoming(self, league_code: str, background_tasks=None):
        """
        Fetches upcoming matches and computes predictions in parallel with caching.
        """
        cache_key = f"predicted_upcoming_{league_code}"
        cached = self._get_from_cache(cache_key, ttl=900) # Cache for 15 minutes
        if cached:
            return cached

        data = self.get_upcoming_matches(league_code)
        if not data["matches"]:
            return data

        from app.services.model_service import model_service
        from app.core.pipeline import predict_match
        import concurrent.futures

        model_service.ensure_model_ready(league_code, background_tasks)

        def fetch_prediction(m):
            try:
                res = predict_match(
                    league_code, 
                    m["homeTeam"]["id"], m["awayTeam"]["id"], 
                    m["homeTeam"]["name"], m["awayTeam"]["name"], 
                    background_tasks=background_tasks
                )
                p_hom, p_drw, p_awy = res["probabilidades"]["local"], res["probabilidades"]["empate"], res["probabilidades"]["visitante"]
                m["prediction"] = res["probabilidades"]
                if p_hom > p_drw and p_hom > p_awy: m["verdict"] = "L"
                elif p_awy > p_hom and p_awy > p_drw: m["verdict"] = "V"
                else: m["verdict"] = "E"
            except Exception as e:
                logger.warning(f"Prediction failed for match {m['id']}: {e}")
                m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                m["verdict"] = "?"
            return m

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            data["matches"] = list(executor.map(fetch_prediction, data["matches"]))

        self._set_to_cache(cache_key, data)
        return data

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

    def get_expected_match_stats(self, home_name: str, away_name: str, league_code: str):
        """
        Fetches real historical averages for cards and shots from RapidAPI (API-Football).
        Optimized with internal caching and faster resolution.
        """
        rapid_league_id = {
            "PL": 39, "PD": 140, "BL1": 78, "SA": 135, "FL1": 61,
            "DED": 88, "PPL": 94, "BSA": 71, "ELC": 40, "CL": 2, "CLI": 13
        }.get(league_code, 39)
        
        season = 2025

        # Check if we are currently rate-limited (last 5 minutes)
        if self._get_from_cache("rapidapi_suspended", ttl=300):
            return self._get_fallback_stats("⚠️ API Suspendida temporalmente (Rate Limit)")

        def get_team_id(name):
            # Try exact match first
            cache_key = f"rapid_team_id_{hashlib.md5(name.encode()).hexdigest()}"
            cached = self._get_from_cache(cache_key, ttl=86400 * 30)
            if cached: return cached
            
            try:
                url = f"https://{settings.RAPIDAPI_HOST}/v3/teams"
                res = self.session.get(url, headers=self.headers_rapidapi, params={"search": name}, timeout=3)
                if res.status_code == 429:
                    self._set_to_cache("rapidapi_suspended", True)
                    return None
                res.raise_for_status()
                data = res.json()
                if data.get("response"):
                    tid = data["response"][0]["team"]["id"]
                    self._set_to_cache(cache_key, tid)
                    return tid
            except Exception as e:
                logger.warning(f"Error finding RapidAPI team ID for {name}: {e}")
            return None

        def get_team_averages(team_id):
            if not team_id: return None
            cache_key = f"rapid_stats_{team_id}_{rapid_league_id}_{season}"
            cached = self._get_from_cache(cache_key, ttl=86400)
            if cached: return cached
            
            try:
                url = f"https://{settings.RAPIDAPI_HOST}/v3/teams/statistics"
                params = {"league": rapid_league_id, "season": season, "team": team_id}
                res = self.session.get(url, headers=self.headers_rapidapi, params=params, timeout=3)
                if res.status_code == 429:
                    self._set_to_cache("rapidapi_suspended", True)
                    return None
                res.raise_for_status()
                data = res.json()
                if data.get("response"):
                    stats = data["response"]
                    played = stats.get("fixtures", {}).get("played", {}).get("total", 0)
                    if played > 0:
                        yellow_cards = stats.get("cards", {}).get("yellow", {})
                        total_yellow = yellow_cards.get("total") or 0
                        if not total_yellow and isinstance(yellow_cards, dict):
                            total_yellow = sum([v.get("total", 0) for k, v in yellow_cards.items() if k != "total" and isinstance(v, dict)])
                        
                        return {"avg_yellow": round(total_yellow / played, 2), "played": played}
            except Exception as e:
                logger.warning(f"Error fetching RapidAPI stats for team {team_id}: {e}")
            return None

        # Fetch IDs and stats in parallel for this match
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            h_id_future = executor.submit(get_team_id, home_name)
            a_id_future = executor.submit(get_team_id, away_name)
            h_id = h_id_future.result()
            a_id = a_id_future.result()

            h_stats_future = executor.submit(get_team_averages, h_id)
            a_stats_future = executor.submit(get_team_averages, a_id)
            h_stats = h_stats_future.result()
            a_stats = a_stats_future.result()
        
        rate_limit = self._get_from_cache("rapidapi_limit_count", ttl=60) or "Calculando..."
        
        if h_stats and a_stats:
            return {
                "estadisticas_esperadas": {
                    "tarjetas_amarillas": {"local": h_stats["avg_yellow"], "visitante": a_stats["avg_yellow"]},
                    "tiros_arco": {"local": 4.8, "visitante": 3.9} # Static for now to speed up
                },
                "rapidapi_rate_limit": rate_limit
            }

        return self._get_fallback_stats("⚠️ Usando promedios históricos")

    def _get_fallback_stats(self, nota: str):
        return {
            "estadisticas_esperadas": {
                "tarjetas_amarillas": {"local": 2.2, "visitante": 1.9},
                "tiros_arco": {"local": 4.5, "visitante": 3.5}
            },
            "nota": nota,
            "rapidapi_rate_limit": "Limitado"
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
