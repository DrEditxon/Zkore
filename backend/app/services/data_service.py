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
            "x-rapidapi-host": settings.RAPIDAPI_HOST,
        }

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

        PERF FIX E: Predictions for the grid skip RapidAPI (include_rapid_stats=False).
        Stats are only fetched when the user opens the detail modal.
        This removes up to 4 external HTTP calls per match from the critical path.

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
                # PERF FIX E: skip RapidAPI for the grid
                res   = predict_match(
                    league_code,
                    m["homeTeam"]["id"], m["awayTeam"]["id"],
                    m["homeTeam"]["name"], m["awayTeam"]["name"],
                    background_tasks=background_tasks,
                    include_rapid_stats=False,
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

    # ─── RapidAPI stats (modal only) ─────────────────────────────────────────

    def get_expected_match_stats(self, home_name: str, away_name: str, league_code: str):
        """
        PERF FIX E: Called only from the detail modal, not from the grid pipeline.
        Dynamic season + circuit-breaker for rate limits.
        """
        rapid_league_id = {
            "PL": 39, "PD": 140, "BL1": 78, "SA": 135, "FL1": 61,
            "DED": 88, "PPL": 94, "BSA": 71, "ELC": 40, "CL": 2, "CLI": 13,
        }.get(league_code, 39)

        from datetime import datetime
        now = datetime.now()
        season = now.year if now.month >= 8 else now.year - 1

        if self._get_from_cache("rapidapi_suspended", ttl=300):
            return self._get_fallback_stats("⚠️ API Suspendida temporalmente (Rate Limit)")

        def get_team_id(name):
            ck = f"rapid_team_id_{hashlib.md5(name.encode()).hexdigest()}"
            hit = self._get_from_cache(ck, ttl=86400 * 30)
            if hit:
                return hit
            try:
                res = self.session.get(
                    f"https://{settings.RAPIDAPI_HOST}/v3/teams",
                    headers=self.headers_rapidapi, params={"search": name}, timeout=3
                )
                if res.status_code == 429:
                    self._set_to_cache("rapidapi_suspended", True)
                    return None
                res.raise_for_status()
                d = res.json()
                if d.get("response"):
                    tid = d["response"][0]["team"]["id"]
                    self._set_to_cache(ck, tid)
                    return tid
            except Exception as e:
                logger.warning(f"RapidAPI team lookup error for {name}: {e}")
            return None

        def get_team_averages(team_id):
            if not team_id:
                return None
            ck = f"rapid_stats_{team_id}_{rapid_league_id}_{season}"
            hit = self._get_from_cache(ck, ttl=86400)
            if hit:
                return hit
            try:
                res = self.session.get(
                    f"https://{settings.RAPIDAPI_HOST}/v3/teams/statistics",
                    headers=self.headers_rapidapi,
                    params={"league": rapid_league_id, "season": season, "team": team_id},
                    timeout=3,
                )
                if res.status_code == 429:
                    self._set_to_cache("rapidapi_suspended", True)
                    return None
                res.raise_for_status()
                d = res.json()
                if d.get("response"):
                    stats  = d["response"]
                    played = stats.get("fixtures", {}).get("played", {}).get("total", 0)
                    if played > 0:
                        yc = stats.get("cards", {}).get("yellow", {})
                        total_y = yc.get("total") or 0
                        if not total_y and isinstance(yc, dict):
                            total_y = sum(
                                v.get("total", 0)
                                for k, v in yc.items()
                                if k != "total" and isinstance(v, dict)
                            )
                        result = {"avg_yellow": round(total_y / played, 2), "played": played}
                        self._set_to_cache(ck, result)
                        return result
            except Exception as e:
                logger.warning(f"RapidAPI stats error for team {team_id}: {e}")
            return None

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            h_id = ex.submit(get_team_id, home_name).result()
            a_id = ex.submit(get_team_id, away_name).result()
            h_stats = ex.submit(get_team_averages, h_id).result()
            a_stats = ex.submit(get_team_averages, a_id).result()

        if h_stats and a_stats:
            return {
                "estadisticas_esperadas": {
                    "tarjetas_amarillas": {
                        "local": h_stats["avg_yellow"],
                        "visitante": a_stats["avg_yellow"],
                    },
                    "tiros_arco": {"local": 4.8, "visitante": 3.9},
                },
                "rapidapi_rate_limit": self._get_from_cache("rapidapi_limit_count", ttl=60) or "OK",
            }

        return self._get_fallback_stats("⚠️ Usando promedios históricos")

    def _get_fallback_stats(self, nota: str):
        return {
            "estadisticas_esperadas": {
                "tarjetas_amarillas": {"local": 2.2, "visitante": 1.9},
                "tiros_arco":        {"local": 4.5, "visitante": 3.5},
            },
            "nota": nota,
            "rapidapi_rate_limit": "Limitado",
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
