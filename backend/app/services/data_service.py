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
            "x-rapidapi-key":  settings.RAPIDAPI_KEY,
            "x-rapidapi-host": settings.RAPIDAPI_HOST,
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
        BUG FIX #1: Catches HTTPException(202) from ensure_model_ready so the page
        never gets stuck on the loading spinner. Returns matches with default
        predictions and a 'training' flag so the frontend can show a banner.
        """
        cache_key = f"predicted_upcoming_{league_code}"
        cached = self._get_from_cache(cache_key, ttl=900)
        if cached:
            return cached

        data = self.get_upcoming_matches(league_code)
        if not data["matches"]:
            return data

        from app.services.model_service import model_service
        from app.core.pipeline import predict_match
        from fastapi import HTTPException
        import concurrent.futures

        # FIX #1a: Capture 202 gracefully — never let it bubble up to the router
        model_ready = False
        try:
            model_service.ensure_model_ready(league_code, background_tasks)
            model_ready = True
        except HTTPException as e:
            if e.status_code == 202:
                logger.info(f"[{league_code}] Model training in progress. Returning matches without predictions.")
                for m in data["matches"]:
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"] = "?"
                    m["training"] = True
                data["training_in_progress"] = True
                data["training_message"] = "Modelo entrenándose, las predicciones estarán listas en ~30 segundos."
                return data
            raise

        if not model_ready:
            for m in data["matches"]:
                m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                m["verdict"] = "?"
            return data

        def fetch_prediction(m):
            try:
                res = predict_match(
                    league_code,
                    m["homeTeam"]["id"], m["awayTeam"]["id"],
                    m["homeTeam"]["name"], m["awayTeam"]["name"],
                    background_tasks=background_tasks
                )
                p_hom = res["probabilidades"]["local"]
                p_drw = res["probabilidades"]["empate"]
                p_awy = res["probabilidades"]["visitante"]
                m["prediction"] = res["probabilidades"]
                if p_hom > p_drw and p_hom > p_awy:
                    m["verdict"] = "L"
                elif p_awy > p_hom and p_awy > p_drw:
                    m["verdict"] = "V"
                else:
                    m["verdict"] = "E"
            except HTTPException as e:
                # Model started training mid-batch — mark gracefully
                if e.status_code == 202:
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"] = "?"
                    m["training"] = True
                else:
                    logger.warning(f"HTTPException for match {m['id']}: {e.detail}")
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"] = "?"
            except Exception as e:
                logger.warning(f"Prediction failed for match {m['id']}: {e}")
                m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                m["verdict"] = "?"
            return m

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            data["matches"] = list(executor.map(fetch_prediction, data["matches"]))

        # Only cache if we got real predictions
        any_training = any(m.get("training") for m in data["matches"])
        if not any_training:
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



    _RAPIDAPI_LEAGUE_MAP = {
        "PL":  {"id": 39,  "season": 2024},
        "PD":  {"id": 140, "season": 2024},
        "BL1": {"id": 78,  "season": 2024},
        "SA":  {"id": 135, "season": 2024},
        "FL1": {"id": 61,  "season": 2024},
        "PPL": {"id": 94,  "season": 2024},
        "DED": {"id": 88,  "season": 2024},
        "BSA": {"id": 71,  "season": 2024},
        "ELC": {"id": 40,  "season": 2024},
    }

    # League-based heuristic fallback (used when RapidAPI is unavailable)
    _HEURISTIC_BASE = {
        "PL":  {"yellow": 1.8, "shots": 4.8},
        "PD":  {"yellow": 2.5, "shots": 4.2},
        "BL1": {"yellow": 2.1, "shots": 4.5},
        "SA":  {"yellow": 2.3, "shots": 4.3},
        "FL1": {"yellow": 2.0, "shots": 4.0},
        "BSA": {"yellow": 2.8, "shots": 3.8},
        "PPL": {"yellow": 2.2, "shots": 4.1},
        "DED": {"yellow": 1.9, "shots": 4.4},
        "ELC": {"yellow": 2.0, "shots": 4.3},
    }

    def _heuristic_stats(self, league_code: str, source: str = "Heurísticas de liga"):
        base = self._HEURISTIC_BASE.get(league_code, {"yellow": 2.1, "shots": 4.2})
        return {
            "estadisticas_esperadas": {
                "tarjetas_amarillas": {
                    "local":     round(base["yellow"] * 0.95, 2),
                    "visitante": round(base["yellow"] * 1.05, 2),
                },
                "tiros_arco": {
                    "local":     round(base["shots"] * 1.10, 2),
                    "visitante": round(base["shots"] * 0.90, 2),
                },
            },
            "nota": source,
            "rapidapi_rate_limit": "N/A",
        }

    def _get_rapidapi_team_id(self, team_name: str, league_id: int, season: int) -> int | None:
        """Search api-football for a team by name within a specific league/season."""
        cache_key = f"rapidapi_team_{league_id}_{season}_{hashlib.md5(team_name.encode()).hexdigest()[:8]}"
        cached = self._get_from_cache(cache_key, ttl=86400 * 7)  # 7-day cache
        if cached is not None:
            return cached

        try:
            url = f"https://{settings.RAPIDAPI_HOST}/teams"
            params = {"name": team_name, "league": league_id, "season": season}
            resp = self.session.get(url, headers=self.headers_rapidapi, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            teams = data.get("response", [])
            if teams:
                team_id = teams[0]["team"]["id"]
                self._set_to_cache(cache_key, team_id)
                return team_id
        except Exception as e:
            logger.warning(f"[RapidAPI] Team lookup failed for '{team_name}': {e}")
        return None

    def _get_rapidapi_team_stats(self, team_id: int, league_id: int, season: int) -> dict | None:
        """Fetch season statistics for a team from api-football."""
        cache_key = f"rapidapi_stats_{team_id}_{league_id}_{season}"
        cached = self._get_from_cache(cache_key, ttl=86400)  # 24h cache
        if cached is not None:
            return cached

        try:
            url = f"https://{settings.RAPIDAPI_HOST}/teams/statistics"
            params = {"team": team_id, "league": league_id, "season": season}
            resp = self.session.get(url, headers=self.headers_rapidapi, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            stats = data.get("response", {})
            if stats:
                self._set_to_cache(cache_key, stats)
                return stats
        except Exception as e:
            logger.warning(f"[RapidAPI] Stats fetch failed for team {team_id}: {e}")
        return None

    def get_expected_match_stats(self, home_name: str, away_name: str, league_code: str) -> dict:
        """
        Fetches real team stats from api-football (RapidAPI) for the current season.
        Falls back gracefully to league heuristics if the key is missing, the API
        is rate-limited, or the request times out.
        """
        if not settings.RAPIDAPI_KEY:
            return self._heuristic_stats(league_code, "Heurísticas de liga (sin clave RapidAPI)")

        league_meta = self._RAPIDAPI_LEAGUE_MAP.get(league_code)
        if not league_meta:
            return self._heuristic_stats(league_code, "Liga no mapeada en RapidAPI")

        league_id = league_meta["id"]
        season    = league_meta["season"]

        try:
            home_id = self._get_rapidapi_team_id(home_name, league_id, season)
            away_id = self._get_rapidapi_team_id(away_name, league_id, season)

            if not home_id or not away_id:
                logger.info(f"[RapidAPI] Team IDs not found for {home_name}/{away_name}, using heuristics")
                return self._heuristic_stats(league_code, "Equipos no encontrados en RapidAPI — usando heurísticas")

            home_stats = self._get_rapidapi_team_stats(home_id, league_id, season)
            away_stats = self._get_rapidapi_team_stats(away_id, league_id, season)

            if not home_stats or not away_stats:
                return self._heuristic_stats(league_code, "Estadísticas no disponibles — usando heurísticas")

            def _safe(stats, *keys, default=0.0):
                val = stats
                for k in keys:
                    val = val.get(k, {}) if isinstance(val, dict) else {}
                return float(val) if isinstance(val, (int, float)) else default

            # Yellow cards per game
            home_yellow_total = _safe(home_stats, "cards", "yellow", "total", default=0)
            away_yellow_total = _safe(away_stats, "cards", "yellow", "total", default=0)
            home_games        = max(1, _safe(home_stats, "fixtures", "played", "total", default=1))
            away_games        = max(1, _safe(away_stats, "fixtures", "played", "total", default=1))
            home_yellow_pg    = round(home_yellow_total / home_games, 2)
            away_yellow_pg    = round(away_yellow_total / away_games, 2)

            # Shots on target per game
            home_shots_total  = _safe(home_stats, "shots", "on", "total", default=0)
            away_shots_total  = _safe(away_stats, "shots", "on", "total", default=0)
            home_shots_pg     = round(home_shots_total / home_games, 2)
            away_shots_pg     = round(away_shots_total / away_games, 2)

            logger.info(f"[RapidAPI] Real stats: {home_name} yel={home_yellow_pg} sot={home_shots_pg} | "
                        f"{away_name} yel={away_yellow_pg} sot={away_shots_pg}")

            return {
                "estadisticas_esperadas": {
                    "tarjetas_amarillas": {
                        "local":     home_yellow_pg,
                        "visitante": away_yellow_pg,
                    },
                    "tiros_arco": {
                        "local":     home_shots_pg,
                        "visitante": away_shots_pg,
                    },
                },
                "nota": f"Estadísticas reales de la temporada {season} (api-football)",
                "rapidapi_rate_limit": "Activo",
            }

        except Exception as e:
            logger.warning(f"[RapidAPI] Unexpected error for {home_name} vs {away_name}: {e}")
            return self._heuristic_stats(league_code, "Error en RapidAPI — usando heurísticas de respaldo")



    def invalidate_cache(self, league_code: str):
        for prefix in ["matches", "standings", "predicted_upcoming"]:
            key = f"{prefix}_{league_code}"
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.warning(f"Error invalidating cache for {key}: {e}")

data_service = DataService()
