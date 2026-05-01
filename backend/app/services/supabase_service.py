import io
import logging
import requests
from app.core.config import settings

logger = logging.getLogger(__name__)

MODELS_BUCKET = "models"


class SupabaseService:
    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.enabled = bool(self.url and self.key)

        if self.enabled:
            # JSON headers for REST API (PostgREST)
            self.headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            }
            # Binary-safe headers for Storage API
            self.storage_headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {self.key}",
            }
            
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            self.session = requests.Session()
            retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504, 429])
            self.session.mount("http://", HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
            self.session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
        else:
            logger.warning("[Supabase] URL or Key not set — integration disabled.")

    # ─── Predictions (REST) ───────────────────────────────────────────────────

    def save_prediction(self, prediction_data: dict):
        """Saves a match prediction to the database (called as a background task)."""
        if not self.enabled:
            return
        try:
            endpoint = f"{self.url}/rest/v1/predictions"
            data = {
                "match_id":    prediction_data["match_id"],
                "league_code": prediction_data["league_code"],
                "home_team":   prediction_data["home_team"],
                "away_team":   prediction_data["away_team"],
                "prediction":  prediction_data["prediction"],
                "prob_home":   prediction_data["prob_home"],
                "prob_draw":   prediction_data["prob_draw"],
                "prob_away":   prediction_data["prob_away"],
                "verdict":     prediction_data["verdict"],
                "utc_date":    prediction_data["utc_date"],
            }
            r = self.session.post(endpoint, headers=self.headers, json=data, timeout=5)
            if r.status_code not in (200, 201, 204):
                logger.error(f"[Supabase] Failed to save prediction: {r.text}")
        except Exception as e:
            logger.error(f"[Supabase] save_prediction error: {e}")

    def get_history(self, league_code: str, limit: int = 10):
        """Fetches recent predictions for a league."""
        if not self.enabled:
            return []
        try:
            endpoint = (
                f"{self.url}/rest/v1/predictions"
                f"?league_code=eq.{league_code}&order=utc_date.desc&limit={limit}"
            )
            r = self.session.get(endpoint, headers=self.headers, timeout=5)
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            logger.error(f"[Supabase] get_history error: {e}")
            return []

    def get_top_predictions(self, limit: int = 10):
        """Fetches top N best predictions globally across all leagues."""
        if not self.enabled:
            return []
        try:
            from datetime import datetime, timezone
            today_str = datetime.now(timezone.utc).isoformat()
            endpoint = f"{self.url}/rest/v1/predictions?utc_date=gte.{today_str}&limit=100"
            r = self.session.get(endpoint, headers=self.headers, timeout=5)
            if r.status_code != 200:
                return []
            predictions = r.json()
            for p in predictions:
                p["max_prob"] = max(
                    p.get("prob_home", 0),
                    p.get("prob_draw", 0),
                    p.get("prob_away", 0),
                )
                if p["max_prob"] == p["prob_home"]:
                    p["predicted_winner"] = p["home_team"]
                elif p["max_prob"] == p["prob_away"]:
                    p["predicted_winner"] = p["away_team"]
                else:
                    p["predicted_winner"] = "Empate"
            predictions.sort(key=lambda x: x["max_prob"], reverse=True)
            return predictions[:limit]
        except Exception as e:
            logger.error(f"[Supabase] get_top_predictions error: {e}")
            return []

    # ─── Feedback Loop: Prediction reconciliation (ML-05) ────────────────────

    def get_unresolved_predictions(self, league_code: str) -> list:
        """
        Returns predictions that have a past utc_date but no actual result yet
        (actual_verdict IS NULL).  Used by FeedbackService to find matches
        to reconcile.
        """
        if not self.enabled:
            return []
        try:
            from datetime import datetime, timezone
            now_str = datetime.now(timezone.utc).isoformat()
            # Fetch predictions whose match date is in the past and have no result
            endpoint = (
                f"{self.url}/rest/v1/predictions"
                f"?league_code=eq.{league_code}"
                f"&utc_date=lt.{now_str}"
                f"&actual_verdict=is.null"
                f"&order=utc_date.desc"
                f"&limit=50"
            )
            r = self.session.get(endpoint, headers=self.headers, timeout=5)
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            logger.error(f"[Supabase] get_unresolved_predictions error: {e}")
            return []

    def resolve_prediction(
        self,
        match_id: int,
        actual_home: int,
        actual_away: int,
        actual_verdict: str,
        brier_score: float,
    ) -> bool:
        """
        Updates a stored prediction row with the real match outcome and the
        computed Brier Score.  Uses PATCH (partial update) so only these fields
        are written; all other prediction data is preserved.

        Returns True on success.  Requires the `predictions` Supabase table to
        have the columns: actual_home_goals, actual_away_goals, actual_verdict,
        brier_score  (all nullable — add them via Supabase dashboard if missing).
        """
        if not self.enabled:
            return False
        try:
            endpoint = f"{self.url}/rest/v1/predictions?match_id=eq.{match_id}"
            patch_headers = {**self.headers, "Content-Type": "application/json"}
            data = {
                "actual_home_goals": actual_home,
                "actual_away_goals": actual_away,
                "actual_verdict":    actual_verdict,
                "brier_score":       round(brier_score, 6),
            }
            r = self.session.patch(endpoint, headers=patch_headers, json=data, timeout=5)
            if r.status_code in (200, 204):
                return True
            logger.error(f"[Supabase] resolve_prediction failed for match {match_id}: {r.text}")
            return False
        except Exception as e:
            logger.error(f"[Supabase] resolve_prediction error: {e}")
            return False

    def get_resolved_predictions(self, league_code: str, days: int = 14) -> list:
        """
        Returns predictions that HAVE been resolved (actual_verdict IS NOT NULL)
        within the last `days` days.  Used by FeedbackService to compute rolling
        Brier Scores for drift detection.
        """
        if not self.enabled:
            return []
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            endpoint = (
                f"{self.url}/rest/v1/predictions"
                f"?league_code=eq.{league_code}"
                f"&utc_date=gte.{cutoff}"
                f"&actual_verdict=not.is.null"
                f"&order=utc_date.desc"
                f"&limit=200"
            )
            r = self.session.get(endpoint, headers=self.headers, timeout=5)
            return r.json() if r.status_code == 200 else []
        except Exception as e:
            logger.error(f"[Supabase] get_resolved_predictions error: {e}")
            return []

    # ─── Historical Matches: Multi-season accumulation (ML-01) ──────────────

    def get_stored_historical_matches(self, league_code: str) -> list:
        """
        ML-01: Fetches all historical matches stored in Supabase for a league.
        Returns them in the same dict format as data_service.get_historical_matches()
        so they can be merged transparently.

        Requires the `historical_matches` table — see SQL in docs.
        """
        if not self.enabled:
            return []
        try:
            endpoint = (
                f"{self.url}/rest/v1/historical_matches"
                f"?league_code=eq.{league_code}"
                f"&order=utc_date.asc"
                f"&limit=10000"
            )
            r = self.session.get(endpoint, headers=self.headers, timeout=10)
            if r.status_code != 200:
                return []
            return [
                {
                    "utcDate":       row["utc_date"],
                    "homeTeam_id":   row["home_team_id"],
                    "homeTeam_name": row["home_team_name"],
                    "awayTeam_id":   row["away_team_id"],
                    "awayTeam_name": row["away_team_name"],
                    "homeGoals":     row["home_goals"],
                    "awayGoals":     row["away_goals"],
                    "match_id":      row["match_id"],
                    "season":        row["season"],
                }
                for row in r.json()
            ]
        except Exception as e:
            logger.error(f"[Supabase] get_stored_historical_matches error: {e}")
            return []

    def store_historical_matches(self, league_code: str, matches: list) -> int:
        """
        ML-01: Bulk-upserts historical match records into Supabase.
        Uses match_id as the unique key — safe to call multiple times (idempotent).
        Batches in chunks of 500 to avoid Supabase payload limits.
        Returns the number of records successfully stored.
        """
        if not self.enabled or not matches:
            return 0
        try:
            rows = [
                {
                    "match_id":       m["match_id"],
                    "league_code":    league_code,
                    "season":         m.get("season"),
                    "utc_date":       m["utcDate"],
                    "home_team_id":   m["homeTeam_id"],
                    "home_team_name": m["homeTeam_name"],
                    "away_team_id":   m["awayTeam_id"],
                    "away_team_name": m["awayTeam_name"],
                    "home_goals":     m["homeGoals"],
                    "away_goals":     m["awayGoals"],
                }
                for m in matches
            ]
            endpoint = f"{self.url}/rest/v1/historical_matches"
            upsert_headers = {
                **self.headers,
                "Content-Type": "application/json",
                "Prefer":       "resolution=merge-duplicates",
            }
            CHUNK = 500
            stored = 0
            for i in range(0, len(rows), CHUNK):
                chunk = rows[i : i + CHUNK]
                r = self.session.post(
                    endpoint, headers=upsert_headers, json=chunk, timeout=30
                )
                if r.status_code in (200, 201, 204):
                    stored += len(chunk)
                else:
                    logger.error(
                        f"[Supabase] store_historical_matches batch {i//CHUNK} "
                        f"failed: {r.status_code} {r.text[:200]}"
                    )
            return stored
        except Exception as e:
            logger.error(f"[Supabase] store_historical_matches error: {e}")
            return 0

    def has_stored_season(self, league_code: str, season: int) -> bool:
        """
        ML-01: Returns True if at least one match for this league+season is
        already stored in Supabase.  Used by the bootstrap to skip re-fetching.
        """
        if not self.enabled:
            return False
        try:
            endpoint = (
                f"{self.url}/rest/v1/historical_matches"
                f"?league_code=eq.{league_code}"
                f"&season=eq.{season}"
                f"&limit=1"
                f"&select=match_id"
            )
            r = self.session.get(endpoint, headers=self.headers, timeout=5)
            return r.status_code == 200 and len(r.json()) > 0
        except Exception as e:
            logger.error(f"[Supabase] has_stored_season error: {e}")
            return False

    # ─── Storage: Model persistence ───────────────────────────────────────────




    def ensure_bucket_exists(self, bucket_name: str = MODELS_BUCKET) -> bool:
        """
        Checks whether the Storage bucket exists and creates it via API if not.
        Idempotent — safe to call on every startup.
        If the Supabase key changes, the bucket is re-provisioned automatically.
        Returns True if the bucket is ready to use.
        """
        if not self.enabled:
            return False
        try:
            r = self.session.get(
                f"{self.url}/storage/v1/bucket/{bucket_name}",
                headers=self.storage_headers,
                timeout=5,
            )
            if r.status_code == 200:
                logger.info(f"[Supabase Storage] Bucket '{bucket_name}' is ready.")
                return True

            # Bucket not found — create it
            logger.info(f"[Supabase Storage] Bucket '{bucket_name}' not found — creating...")
            r2 = self.session.post(
                f"{self.url}/storage/v1/bucket",
                headers={**self.storage_headers, "Content-Type": "application/json"},
                json={"id": bucket_name, "name": bucket_name, "public": False},
                timeout=5,
            )
            if r2.status_code in (200, 201):
                logger.info(f"[Supabase Storage] Bucket '{bucket_name}' created successfully.")
                return True

            logger.error(
                f"[Supabase Storage] Failed to create bucket '{bucket_name}': "
                f"{r2.status_code} {r2.text}"
            )
            return False
        except Exception as e:
            logger.error(f"[Supabase Storage] ensure_bucket_exists error: {e}")
            return False

    def upload_model(self, league_code: str, model_bytes: bytes, bucket: str = MODELS_BUCKET) -> bool:
        """
        Uploads a serialized model (.joblib bytes) to Supabase Storage.
        Uses x-upsert so re-training always replaces the previous version.
        Returns True on success.
        """
        if not self.enabled:
            return False
        filename = f"{league_code}_xgb_latest.joblib"
        try:
            r = self.session.post(
                f"{self.url}/storage/v1/object/{bucket}/{filename}",
                headers={
                    **self.storage_headers,
                    "Content-Type": "application/octet-stream",
                    "x-upsert": "true",
                },
                data=model_bytes,
                timeout=60,
            )
            if r.status_code in (200, 201):
                logger.info(
                    f"[Supabase Storage] Uploaded '{filename}' ({len(model_bytes):,} bytes)."
                )
                return True
            logger.error(
                f"[Supabase Storage] Upload failed for '{filename}': {r.status_code} {r.text}"
            )
            return False
        except Exception as e:
            logger.error(f"[Supabase Storage] upload_model error: {e}")
            return False

    def download_model(self, league_code: str, bucket: str = MODELS_BUCKET) -> bytes | None:
        """
        Downloads a model file from Supabase Storage.
        Returns raw bytes on success, None if not found or on error.
        """
        if not self.enabled:
            return None
        filename = f"{league_code}_xgb_latest.joblib"
        try:
            r = self.session.get(
                f"{self.url}/storage/v1/object/{bucket}/{filename}",
                headers=self.storage_headers,
                timeout=60,
            )
            if r.status_code == 200:
                logger.info(
                    f"[Supabase Storage] Downloaded '{filename}' ({len(r.content):,} bytes)."
                )
                return r.content
            if r.status_code == 404:
                logger.info(f"[Supabase Storage] '{filename}' not found in bucket.")
                return None
            logger.error(
                f"[Supabase Storage] Download failed for '{filename}': {r.status_code} {r.text}"
            )
            return None
        except Exception as e:
            logger.error(f"[Supabase Storage] download_model error: {e}")
            return None

    def list_models(self, bucket: str = MODELS_BUCKET) -> list:
        """
        Returns a list of model file objects stored in the bucket.
        Each item has at least a 'name' key.
        """
        if not self.enabled:
            return []
        try:
            r = self.session.post(
                f"{self.url}/storage/v1/object/list/{bucket}",
                headers={**self.storage_headers, "Content-Type": "application/json"},
                json={"prefix": "", "limit": 200},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            logger.error(f"[Supabase Storage] list_models failed: {r.status_code} {r.text}")
            return []
        except Exception as e:
            logger.error(f"[Supabase Storage] list_models error: {e}")
            return []


supabase_service = SupabaseService()
