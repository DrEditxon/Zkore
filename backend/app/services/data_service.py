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

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504, 429])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # PERF FIX: In-memory cache for historical matches — avoids JSON deserialization
        # on every predict call. Invalidated when fresh data is fetched.
        self._matches_mem_cache: dict = {}

    # ─── Disk cache helpers ───────────────────────────────────────────────────

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

    # ─── Leagues ─────────────────────────────────────────────────────────────

    def get_leagues(self):
        return [
            {"code": code, "name": meta["name"], "flag": meta["flag"]}
            for code, meta in settings.LEAGUES_METADATA.items()
        ]

    # ─── Upcoming matches ────────────────────────────────────────────────────

    def get_upcoming_matches(self, league_code: str, limit: int = 10) -> dict:
        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        try:
            response = self.session.get(
                url, headers=self.headers_football_data,
                params={"status": "SCHEDULED"}, timeout=10
            )
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
                    "crest": m["homeTeam"]["crest"],
                },
                "awayTeam": {
                    "id": m["awayTeam"]["id"],
                    "name": m["awayTeam"]["name"],
                    "crest": m["awayTeam"]["crest"],
                },
            })
        return {"matchday": current_matchday, "matches": matches}

    def get_predicted_upcoming(self, league_code: str, background_tasks=None):
        """
        Fetches upcoming matches and computes predictions in parallel.

        BUG FIX #1: Catches HTTPException(202) so the page never gets stuck loading.
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

        model_ready = False
        try:
            model_service.ensure_model_ready(league_code, background_tasks)
            model_ready = True
        except HTTPException as e:
            if e.status_code == 202:
                logger.info(f"[{league_code}] Training in progress — returning unscored grid.")
                for m in data["matches"]:
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"]    = "?"
                    m["training"]   = True
                data["training_in_progress"] = True
                data["training_message"] = "Modelo entrenándose, las predicciones estarán listas en ~30 segundos."
                return data
            raise

        if not model_ready:
            for m in data["matches"]:
                m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                m["verdict"]    = "?"
            return data

        def fetch_prediction(m):
            try:
                res   = predict_match(
                    league_code,
                    m["homeTeam"]["id"], m["awayTeam"]["id"],
                    m["homeTeam"]["name"], m["awayTeam"]["name"],
                    background_tasks=background_tasks,
                    match_id=m["id"],
                    utc_date=m["utcDate"],
                )
                p_hom = res["probabilidades"]["local"]
                p_drw = res["probabilidades"]["empate"]
                p_awy = res["probabilidades"]["visitante"]
                m["prediction"] = res["probabilidades"]
                if p_hom > p_drw and p_hom > p_awy:     m["verdict"] = "L"
                elif p_awy > p_hom and p_awy > p_drw:   m["verdict"] = "V"
                else:                                     m["verdict"] = "E"
            except HTTPException as e:
                if e.status_code == 202:
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"]    = "?"
                    m["training"]   = True
                else:
                    logger.warning(f"HTTPException for match {m['id']}: {e.detail}")
                    m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                    m["verdict"]    = "?"
            except Exception as e:
                logger.warning(f"Prediction failed for match {m['id']}: {e}")
                m["prediction"] = {"local": 33.3, "empate": 33.3, "visitante": 33.3}
                m["verdict"]    = "?"
            return m

        # PERF: keep max_workers=5 — more workers than this hits GIL contention
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            data["matches"] = list(executor.map(fetch_prediction, data["matches"]))

        any_training = any(m.get("training") for m in data["matches"])
        if not any_training:
            self._set_to_cache(cache_key, data)

        return data

    # ─── Standings ───────────────────────────────────────────────────────────

    def get_standings(self, league_code: str):
        cached = self._get_from_cache(f"standings_{league_code}")
        if cached:
            return cached
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

    # ─── Historical matches ──────────────────────────────────────────────────

    def get_historical_matches(self, league_code: str) -> list:
        """
        PERF FIX: Two-level cache.
        1. In-memory (_matches_mem_cache): instant — avoids JSON parse on every predict call.
        2. Disk: survives process restart within the TTL window.
        """
        if league_code in self._matches_mem_cache:
            return self._matches_mem_cache[league_code]

        cached = self._get_from_cache(f"matches_{league_code}", ttl=settings.CACHE_DURATION_HISTORICAL)
        if cached:
            self._matches_mem_cache[league_code] = cached
            return cached

        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        try:
            response = self.session.get(
                url, headers=self.headers_football_data,
                params={"status": "FINISHED"}, timeout=15
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical matches for {league_code}: {e}")
            return []

        data = response.json()
        matches = []
        for m in data.get("matches", []):
            score     = m.get("score", {})
            full      = score.get("fullTime", {})
            home_goals = full.get("home")
            away_goals = full.get("away")
            if home_goals is None or away_goals is None:
                continue
            matches.append({
                "utcDate":        m["utcDate"],
                "homeTeam_id":    m["homeTeam"]["id"],
                "homeTeam_name":  m["homeTeam"]["name"],
                "awayTeam_id":    m["awayTeam"]["id"],
                "awayTeam_name":  m["awayTeam"]["name"],
                "homeGoals":      home_goals,
                "awayGoals":      away_goals,
            })

        self._set_to_cache(f"matches_{league_code}", matches)
        self._matches_mem_cache[league_code] = matches
        return matches

    # ─── Modal Stats ─────────────────────────────────────────

    def get_expected_match_stats(self, home_name: str, away_name: str, league_code: str):
        """
        RapidAPI integration has been removed for simplicity and stability.
        Returns static historical baseline stats to maintain frontend compatibility.
        """
        return {
            "estadisticas_esperadas": {
                "tarjetas_amarillas": {"local": 2.2, "visitante": 1.9},
                "tiros_arco":        {"local": 4.5, "visitante": 3.5},
            },
            "nota": "Estadísticas predictivas basales activadas.",
        }

    # ─── Cache invalidation ───────────────────────────────────────────────────

    def invalidate_cache(self, league_code: str):
        self._matches_mem_cache.pop(league_code, None)
        from app.services.feature_service import feature_service
        feature_service.invalidate_cache()
        for prefix in ["matches", "standings", "predicted_upcoming"]:
            key = f"{prefix}_{league_code}"
            path = self._get_cache_path(key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Error invalidating cache for {key}: {e}")


data_service = DataService()
