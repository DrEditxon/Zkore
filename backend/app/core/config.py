import os

class Settings:
    # API Keys
    FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "e4b22b463e054ef59664fd74fc3f94dd")
    RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "ffcdab306fmshe3b6e9b4cd55ed4p137d4ejsn3d598ba17cff")
    RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
    
    # Cache duration in seconds (1 hour default)
    CACHE_DURATION = int(os.getenv("CACHE_DURATION", "3600"))
    
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
