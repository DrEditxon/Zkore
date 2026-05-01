"""
BUG D FIX: Scheduled background retraining.

Without this, models only retrain when a user hits /upcoming.
On Render Free Tier the service sleeps after inactivity — models
can stay stale for weeks if no one visits.

This scheduler runs once at startup and checks every 6 hours whether
any league model is stale (> 7 days) or missing, and retrains it.
Uses only stdlib threading — no extra dependencies (no APScheduler/Celery).
"""

import threading
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# How often to check for stale models (seconds)
SCHEDULE_INTERVAL_SECONDS = 6 * 3600  # 6 hours

# Leagues to keep warm proactively
# Matches config.py LEAGUES_METADATA keys (minus CL/CLI which have no models)
SCHEDULED_LEAGUES = ["PL", "PD", "BL1", "SA", "FL1", "PPL", "DED", "BSA", "ELC"]


def _run_scheduled_retraining():
    """
    Runs in a background daemon thread.
    Every 6 hours, checks each league and retrains if model is stale or missing.
    Also refreshes the historical data cache (invalidate → re-fetch) before training
    so models always incorporate the latest finished matches.
    """
    # Wait 90 seconds at startup to let the server fully initialize
    time.sleep(90)
    logger.info("[Scheduler] Background retraining scheduler started.")

    while True:
        logger.info(f"[Scheduler] Running scheduled model check at {datetime.now().isoformat()}")

        try:
            # Import here to avoid circular imports at module load time
            from app.services.model_service import model_service
            from app.services.data_service import data_service

            for league_code in SCHEDULED_LEAGUES:
                try:
                    payload = model_service._load_model(league_code)

                    if payload is None:
                        logger.info(f"[Scheduler] {league_code}: No model found — triggering training.")
                        _retrain_league(league_code, data_service, model_service)

                    elif payload.get("is_stale", False):
                        age = payload.get("model_age_days", "?")
                        logger.info(f"[Scheduler] {league_code}: Model is {age} days old — retraining.")
                        _retrain_league(league_code, data_service, model_service)

                    else:
                        age = payload.get("model_age_days", "?")
                        logger.info(f"[Scheduler] {league_code}: Model is fresh ({age} days). Skipping.")

                except Exception as e:
                    logger.error(f"[Scheduler] Error checking {league_code}: {e}")

        except Exception as e:
            logger.error(f"[Scheduler] Unexpected error in scheduling loop: {e}")

        # ML-05: Run feedback loop for all leagues (separate try/except so a
        # Supabase error never blocks the normal model-check cycle)
        try:
            from app.services.feedback_service import feedback_service
            logger.info("[Scheduler] Running feedback loop (prediction reconciliation)...")
            for league_code in SCHEDULED_LEAGUES:
                try:
                    summary = feedback_service.run_for_league(league_code)
                    if summary["resolved"] > 0 or summary["avg_brier"] is not None:
                        logger.info(
                            f"[Scheduler] [{league_code}] Feedback: "
                            f"resolved={summary['resolved']}, "
                            f"brier={summary['avg_brier']}, "
                            f"drift={summary['drift_detected']}"
                        )
                except Exception as e:
                    logger.error(f"[Scheduler] Feedback error for {league_code}: {e}")
        except Exception as e:
            logger.error(f"[Scheduler] Feedback loop import/run error: {e}")

        logger.info(f"[Scheduler] Next check in {SCHEDULE_INTERVAL_SECONDS // 3600} hours.")
        time.sleep(SCHEDULE_INTERVAL_SECONDS)


def _retrain_league(league_code: str, data_service, model_service):
    """Synchronously retrain a single league inside the scheduler thread."""
    # BUG-03 FIX: Use _is_training_locked() instead of the in-memory dict so
    # that the scheduler correctly detects training started by any Gunicorn worker,
    # not just the one running this scheduler thread.
    if model_service._is_training_locked(league_code):
        logger.info(f"[Scheduler] {league_code}: Training already in progress (cross-process check). Skipping.")
        return

    model_service._training_in_progress[league_code] = True
    try:
        # BUG B FIX: Invalidate cache so we pull the latest results from the API
        data_service.invalidate_cache(league_code)
        matches = data_service.get_historical_matches(league_code)

        if len(matches) < 10:
            logger.warning(f"[Scheduler] {league_code}: Only {len(matches)} matches — skipping.")
            return

        model_service._train_and_save(league_code, matches)
        logger.info(f"[Scheduler] {league_code}: Retraining complete ✓")

    except Exception as e:
        logger.error(f"[Scheduler] {league_code}: Retraining failed: {e}")
    finally:
        model_service._training_in_progress[league_code] = False


def start_scheduler():
    """
    Launch the retraining scheduler as a daemon thread.
    Called once from main.py @app startup event.
    Daemon=True ensures the thread dies when the main process exits.
    """
    t = threading.Thread(target=_run_scheduled_retraining, daemon=True, name="ZkoreScheduler")
    t.start()
    logger.info("[Scheduler] Daemon thread launched.")
