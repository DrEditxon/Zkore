import os
from dotenv import load_dotenv

# Load .env from the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"), override=True)

class Settings:
    # API Keys
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
    if not FOOTBALL_DATA_API_KEY:
        raise ValueError("FOOTBALL_DATA_API_KEY is required but not set in the environment.")
        
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
    if not RAPIDAPI_KEY:
        raise ValueError("RAPIDAPI_KEY is required but not set in the environment.")
        
    RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
    
    # Cache duration in seconds
    CACHE_DURATION = int(os.getenv("CACHE_DURATION", "3600"))
    CACHE_DURATION_HISTORICAL = int(os.getenv("CACHE_DURATION_HISTORICAL", "86400"))
    
    # Security
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    
    # Model configuration
    MIN_MATCHES_REQUIRED = 30
    
    # Supported Leagues
    LEAGUES_METADATA = {
        "PL": {"name": "Premier League", "flag": "https://crests.football-data.org/770.svg"},
        "PD": {"name": "La Liga", "flag": "https://crests.football-data.org/760.svg"},
        "BL1": {"name": "Bundesliga", "flag": "https://crests.football-data.org/759.svg"},
        "SA": {"name": "Serie A", "flag": "https://crests.football-data.org/784.svg"},
        "FL1": {"name": "Ligue 1", "flag": "https://crests.football-data.org/773.svg"},
        "PPL": {"name": "Primeira Liga", "flag": "https://crests.football-data.org/765.svg"},
        "DED": {"name": "Eredivisie", "flag": "https://crests.football-data.org/8601.svg"},
        "BSA": {"name": "Série A", "flag": "https://crests.football-data.org/764.svg"},
        "ELC": {"name": "Championship", "flag": "https://crests.football-data.org/770.svg"},
        "CL": {"name": "UEFA Champions League", "flag": "https://crests.football-data.org/EUR.svg"},
        "CLI": {"name": "Copa Libertadores", "flag": "https://crests.football-data.org/CLI.svg"}
    }

settings = Settings()
