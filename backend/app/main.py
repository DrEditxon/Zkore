import logging
import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import concurrent.futures

from app.core.config import settings
from app.core.limiter import limiter

from app.routes.predict import router as predict_router
from app.services.data_service import data_service
from app.services.model_service import model_service
from app.services.poisson_service import poisson_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Zkore Prediction API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Includes /predict endpoint
app.include_router(predict_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API running cleanly"}

# Fix static files path for running from backend/ directory
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

try:
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
except RuntimeError:
    logger.warning("Frontend directory not found, static files will not be mounted.")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/leagues")
def list_leagues():
    return data_service.get_leagues()

@app.get("/upcoming/{league_code}")
def list_upcoming(league_code: str, background_tasks: BackgroundTasks):
    return data_service.get_predicted_upcoming(league_code, background_tasks)

