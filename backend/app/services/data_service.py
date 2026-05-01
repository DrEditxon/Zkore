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

        # ── Enrich with live data from ESPN ─────────────────────────────
        try:
            from app.services.market_service import market_service
            import difflib
            live_data = market_service.get_live_matches(league_code)
            
            for m in data["matches"]:
                h_norm = market_service._normalize_name(m["homeTeam"]["name"])
                a_norm = market_service._normalize_name(m["awayTeam"]["name"])
                live_info = live_data.get((h_norm, a_norm))
                
                if not live_info:
                    for (hk, ak), v in live_data.items():
                        if (difflib.SequenceMatcher(None, h_norm, hk).ratio() > 0.7 or h_norm in hk or hk in h_norm) and \
                           (difflib.SequenceMatcher(None, a_norm, ak).ratio() > 0.7 or a_norm in ak or ak in a_norm):
                            live_info = v
                            break
                            
                if live_info:
                    m["is_live"] = live_info["state"] == "in"
                    m["live_score"] = live_info["score"]
                    m["live_minute"] = live_info["minute"]
                    m["status_espn"] = live_info["state"]
        except Exception as e:
            logger.error(f"Failed to enrich live matches: {e}")

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

    # ─── Multi-season data accumulation (ML-01) ──────────────────────────────

    def _fetch_season(self, league_code: str, season_year: int) -> list:
        """
        ML-01: Fetches all FINISHED matches for one past season from
        football-data.org using the `?season=YYYY` query parameter.

        Returns a list of match dicts in the same format as get_historical_matches()
        plus `match_id` (int) and `season` (int) fields needed for Supabase storage.
        """
        import requests as req_mod
        url = "https://api.football-data.org/v4/competitions/{code}/matches".format(
            code=league_code
        )
        try:
            r = self.session.get(
                url,
                headers={"X-Auth-Token": self._football_data_key()},
                params={"status": "FINISHED", "season": season_year},
                timeout=20,
            )
            if r.status_code == 404:
                logger.warning(f"[{league_code}] Season {season_year} not found (404).")
                return []
            if r.status_code == 429:
                logger.warning(f"[{league_code}] Rate limited fetching season {season_year}.")
                return []
            r.raise_for_status()
        except Exception as e:
            logger.error(f"[{league_code}] Error fetching season {season_year}: {e}")
            return []

        matches = []
        for m in r.json().get("matches", []):
            score = m.get("score", {})
            full  = score.get("fullTime", {})
            hg, ag = full.get("home"), full.get("away")
            if hg is None or ag is None:
                continue
            matches.append({
                "match_id":      m["id"],
                "season":        season_year,
                "utcDate":       m["utcDate"],
                "homeTeam_id":   m["homeTeam"]["id"],
                "homeTeam_name": m["homeTeam"]["name"],
                "awayTeam_id":   m["awayTeam"]["id"],
                "awayTeam_name": m["awayTeam"]["name"],
                "homeGoals":     hg,
                "awayGoals":     ag,
            })
        return matches

    def _football_data_key(self) -> str:
        """Helper to extract the API key from the stored session headers."""
        return self.session.headers.get("X-Auth-Token", "")

    def get_historical_matches_multi_season(self, league_code: str) -> list:
        """
        ML-01 FIX: Returns ALL available matches for a league — past seasons
        from Supabase merged with the current live season from the API.

        Deduplication key: (utcDate[:10], homeTeam_id, awayTeam_id).
        Current-season entry wins on conflicts (most accurate data).

        Falls back to single-season if Supabase is disabled or the table is
        empty (first deploy before bootstrap completes).
        """
        from app.services.supabase_service import supabase_service

        current = self.get_historical_matches(league_code)

        if not supabase_service.enabled:
            return current  # Graceful degradation

        stored = supabase_service.get_stored_historical_matches(league_code)
        if not stored:
            logger.debug(f"[{league_code}] No stored historical matches yet — using current season only.")
            return current

        # Merge: build lookup from stored, then overwrite with current (authoritative)
        all_matches: dict = {}
        for m in stored:
            key = (m["utcDate"][:10], int(m["homeTeam_id"]), int(m["awayTeam_id"]))
            all_matches[key] = m
        for m in current:
            key = (m["utcDate"][:10], int(m["homeTeam_id"]), int(m["awayTeam_id"]))
            all_matches[key] = m  # current season always wins

        merged = sorted(all_matches.values(), key=lambda x: x["utcDate"])
        logger.info(
            f"[{league_code}] Multi-season merge: {len(stored)} stored + "
            f"{len(current)} current = {len(merged)} total"
        )
        return merged

    def bootstrap_historical_seasons(
        self, league_code: str, seasons_back: int = 3
    ) -> None:
        """
        ML-01: One-time background bootstrap to fetch and store past seasons.

        Design decisions:
        ─────────────────
        - Idempotent: skips seasons already in Supabase (safe to call every startup).
        - Rate-limited: 7s sleep between API calls to respect football-data.org
          free tier (10 req/min).
        - Fire-and-forget: runs in a daemon thread; failures are logged, not raised.
        - Season calculation: seasons are identified by starting year (e.g., 2023
          means the 2023/24 season). If current month < August, current season
          started the previous calendar year.
        """
        import time
        from app.services.supabase_service import supabase_service
        from datetime import datetime

        if not supabase_service.enabled:
            logger.info(f"[{league_code}] Supabase disabled — skipping historical bootstrap.")
            return

        now = datetime.now()
        # Football seasons start in August; before Aug the current season is (year-1)
        current_season = now.year if now.month >= 8 else now.year - 1
        past_seasons   = [current_season - i for i in range(1, seasons_back + 1)]

        for season_year in past_seasons:
            try:
                if supabase_service.has_stored_season(league_code, season_year):
                    logger.info(
                        f"[{league_code}] Season {season_year} already stored — skipping."
                    )
                    continue

                logger.info(f"[{league_code}] Bootstrapping season {season_year}...")
                matches = self._fetch_season(league_code, season_year)

                if not matches:
                    logger.warning(
                        f"[{league_code}] Season {season_year}: no matches returned."
                    )
                    time.sleep(7)
                    continue

                count = supabase_service.store_historical_matches(league_code, matches)
                logger.info(
                    f"[{league_code}] Season {season_year}: stored {count}/{len(matches)} matches."
                )

                # Rate limiting: stay safely under 10 req/min
                time.sleep(7)

            except Exception as e:
                logger.error(f"[{league_code}] Bootstrap error for season {season_year}: {e}")

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
