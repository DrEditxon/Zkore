import requests
import time

API_KEY = "e4b22b463e054ef59664fd74fc3f94dd"
headers = {"X-Auth-Token": API_KEY}

# Supported Free-Tier Domestic Leagues with visual metadata
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

# Simple Time-based Cache
cache = {}
CACHE_DURATION = 3600  # 1 hour in seconds

def get_from_cache(key):
    if key in cache:
        entry, timestamp = cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            return entry
    return None

def set_to_cache(key, data):
    cache[key] = (data, time.time())

def get_leagues():
    return [{"code": code, "name": meta["name"], "flag": meta["flag"]} for code, meta in LEAGUES_METADATA.items()]

def get_upcoming_matches(league_code: str, limit: int = 10) -> dict:
    """
    Fetches scheduled matches for the competition.
    Returns {matchday: N, matches: [...]}
    """
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    params = {"status": "SCHEDULED"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
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
                "crest": m["homeTeam"]["crest"]
            },
            "awayTeam": {
                "id": m["awayTeam"]["id"],
                "name": m["awayTeam"]["name"],
                "crest": m["awayTeam"]["crest"]
            }
        })
    
    return {"matchday": current_matchday, "matches": matches}

def get_standings(league_code):
    cached_data = get_from_cache(f"standings_{league_code}")
    if cached_data:
        return cached_data

    url = f"https://api.football-data.org/v4/competitions/{league_code}/standings"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    
    data = response.json()
    set_to_cache(f"standings_{league_code}", data)
    return data

def calculate_league_averages(standings_data):
    # We use the 'TOTAL' table to calculate averages
    table = standings_data['standings'][0]['table']
    total_goals = sum(team['goalsFor'] for team in table)
    total_matches = sum(team['playedGames'] for team in table)
    
    # Each match has 2 teams, so total goals / (total_matches / 2 * 2) ? 
    # Actually league avg goals per team per match = total_goals / (total_matches)
    if total_matches == 0:
        return 1.3, 1.3 # Fallback
    
    avg_goals = total_goals / total_matches
    return avg_goals, avg_goals # Simplified for now, or we could fetch Home/Away specifically

def get_team_relative_stats(league_code, team_id):
    standings = get_standings(league_code)
    if not standings:
        return None
    
    avg_goals, _ = calculate_league_averages(standings)
    table = standings['standings'][0]['table']
    
    team_data = next((team for team in table if team['team']['id'] == team_id), None)
    if not team_data or team_data['playedGames'] == 0:
        return 1.0, 1.0, avg_goals
    
    # Attack Strength = (Goals Scored / Games) / League Avg
    attack_strength = (team_data['goalsFor'] / team_data['playedGames']) / avg_goals
    # Defense Strength = (Goals Conceded / Games) / League Avg
    defense_strength = (team_data['goalsAgainst'] / team_data['playedGames']) / avg_goals
    
    return attack_strength, defense_strength, avg_goals


def get_historical_matches(league_code: str) -> list:
    """
    Fetches all FINISHED matches for the current season of a league.
    Returns a flat list of match dicts with normalized fields.
    """
    cached = get_from_cache(f"matches_{league_code}")
    if cached:
        return cached

    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    params = {"status": "FINISHED"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return []

    data = response.json()
    matches = []
    for m in data.get("matches", []):
        score = m.get("score", {})
        full = score.get("fullTime", {})
        home_goals = full.get("home")
        away_goals = full.get("away")
        if home_goals is None or away_goals is None:
            continue
        matches.append({
            "utcDate": m["utcDate"],
            "homeTeam_id": m["homeTeam"]["id"],
            "homeTeam_name": m["homeTeam"]["name"],
            "awayTeam_id": m["awayTeam"]["id"],
            "awayTeam_name": m["awayTeam"]["name"],
            "homeGoals": home_goals,
            "awayGoals": away_goals,
        })

    set_to_cache(f"matches_{league_code}", matches)
    return matches