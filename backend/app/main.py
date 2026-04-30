import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.limiter import limiter
from app.core.scheduler import start_scheduler          # BUG D FIX

from app.routes.predict import router as predict_router
from app.services.data_service import data_service
from app.services.supabase_service import supabase_service
from app.services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: runs on startup and shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Zkore Prediction API...")
    start_scheduler()
    logger.info("Scheduled retraining daemon started (every 6 hours).")

    # ── Supabase Storage boot sequence ──────────────────────────────────────
    # Runs in a thread so it never blocks the HTTP server from accepting requests.
    # - Creates the 'models' bucket via API if it doesn't exist yet.
    # - Migrates existing local .joblib files to Supabase (skips already-uploaded ones).
    # If the Supabase key ever changes, missing models are re-uploaded automatically.
    def _supabase_boot():
        try:
            bucket_ready = supabase_service.ensure_bucket_exists()
            if bucket_ready:
                model_service.migrate_local_models_to_supabase()
        except Exception as e:
            logger.error(f"[Supabase Boot] Non-fatal error: {e}")

    import threading
    threading.Thread(target=_supabase_boot, daemon=True, name="supabase-boot").start()

    yield
    logger.info("Zkore API shutting down.")


app = FastAPI(title="Zkore Prediction API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    from app.services.model_service import model_service
    import glob

    model_files = {}
    for league in settings.LEAGUES_METADATA:
        files = glob.glob(
            os.path.join(
                os.path.dirname(__file__), "models", f"{league}_xgb_*.joblib"
            )
        )
        payload = model_service._model_cache.get(league)
        model_files[league] = {
            "files_on_disk": len(files),
            "loaded_in_cache": payload is not None,
            "age_days": payload.get("model_age_days") if payload else None,
            "is_stale": payload.get("is_stale") if payload else None,
            "training_in_progress": model_service._training_in_progress.get(league, False),
        }

    return {
        "status": "ok",
        "models": model_files,
    }


# ── Static / Frontend ─────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
)

try:
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    logger.info(f"Frontend served from: {FRONTEND_DIR}")
except RuntimeError:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}. Static files disabled.")


@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/leagues")
def list_leagues():
    return data_service.get_leagues()


@app.get("/upcoming/{league_code}")
def list_upcoming(league_code: str, background_tasks: BackgroundTasks):
    return data_service.get_predicted_upcoming(league_code, background_tasks)
