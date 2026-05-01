import os
from dotenv import load_dotenv

# Load .env from the root directory (only relevant for local dev; Render uses env vars)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"), override=True)

class Settings:
    # API Keys
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    if not FOOTBALL_DATA_API_KEY:
        raise ValueError("FOOTBALL_DATA_API_KEY is required but not set in the environment.")

    # Cache duration in seconds
    CACHE_DURATION = int(os.getenv("CACHE_DURATION", "3600"))
    CACHE_DURATION_HISTORICAL = int(os.getenv("CACHE_DURATION_HISTORICAL", "86400"))

    # FIX #4: CORS default now uses "*" so the app works on any deployment
    # without requiring ALLOWED_ORIGINS to be manually configured.
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Security Configuration
    ENFORCE_API_KEY = os.getenv("ENFORCE_API_KEY", "False").lower() in ("true", "1", "yes")
    ZKORE_API_KEY = os.getenv("ZKORE_API_KEY", "zkore-secret-dev-key")

    # Model configuration
    MIN_MATCHES_REQUIRED = int(os.getenv("MIN_MATCHES_REQUIRED", "30"))

    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    # FIX #10: Removed CL and CLI — no trained models exist for them.
    # Add them back once models are trained and included in backend/app/models/.
    LEAGUES_METADATA = {
        "PL":  {"name": "Premier League",   "flag": "https://crests.football-data.org/770.svg"},
        "PD":  {"name": "La Liga",           "flag": "https://crests.football-data.org/760.svg"},
        "BL1": {"name": "Bundesliga",        "flag": "https://crests.football-data.org/759.svg"},
        "SA":  {"name": "Serie A",           "flag": "https://crests.football-data.org/784.svg"},
        "FL1": {"name": "Ligue 1",           "flag": "https://crests.football-data.org/773.svg"},
        "PPL": {"name": "Primeira Liga",     "flag": "https://crests.football-data.org/765.svg"},
        "DED": {"name": "Eredivisie",        "flag": "https://crests.football-data.org/8601.svg"},
        "BSA": {"name": "Série A Brasil",    "flag": "https://crests.football-data.org/764.svg"},
        "ELC": {"name": "Championship",      "flag": "https://crests.football-data.org/770.svg"},
    }

settings = Settings()
