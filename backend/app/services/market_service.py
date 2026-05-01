import logging
import requests
from datetime import datetime, timezone
import difflib
import json
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

# Mapping from football-data.org league codes to The Odds API sport keys
LEAGUE_MAP = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "PPL": "soccer_portugal_primeira_liga",
    "DED": "soccer_netherlands_eredivisie",
    "BSA": "soccer_brazil_campeonato",
    "ELC": "soccer_efl_championship",
}

# Mapping to ESPN API league codes
ESPN_MAP = {
    "PL": "eng.1",
    "PD": "esp.1",
    "BL1": "ger.1",
    "SA": "ita.1",
    "FL1": "fra.1",
    "PPL": "por.1",
    "DED": "ned.1",
    "BSA": "bra.1",
    "ELC": "eng.2",
    "COL": "col.1",
}


CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")

class MarketService:
    """
    Integrates with The Odds API to fetch live bookmaker odds.
    Converts them to true implied probabilities (removing overround/vig)
    so they can be ensembled with the XGBoost native predictions.
    """
    def __init__(self):
        self.api_key = settings.ODDS_API_KEY
        self.enabled = bool(self.api_key)
        self.session = requests.Session()
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        if not self.enabled:
            logger.info("The Odds API key not configured. Market ensemble disabled.")

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace("fc ", "").replace(" fc", "").strip()

    def _get_cache_path(self, sport_key: str, prefix: str = "odds_") -> str:
        return os.path.join(CACHE_DIR, f"{prefix}{sport_key}.json")

    def _american_to_decimal(self, american_str: str) -> float:
        val = str(american_str).strip().upper()
        if val == "EVEN":
            return 2.0
        try:
            if val.startswith('+'):
                return (float(val[1:]) / 100.0) + 1.0
            elif val.startswith('-'):
                return (100.0 / float(val[1:])) + 1.0
            else:
                return float(val)
        except ValueError:
            return 0.0

    def _fetch_espn_odds(self, espn_code: str, home_name: str, away_name: str) -> dict | None:
        """
        Fetches odds from ESPN's free scoreboard API (DraftKings odds).
        Returns true implied probabilities if found, else None.
        """
        cache_path = self._get_cache_path(espn_code, prefix="espn_")
        
        # 15 minute cache for ESPN
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if datetime.now().timestamp() - mtime < 900:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = None
            else:
                data = None
        else:
            data = None

        if not data:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard"
                r = self.session.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                else:
                    return None
            except Exception as e:
                logger.error(f"ESPN API error for {espn_code}: {e}")
                return None

        norm_home = self._normalize_name(home_name)
        norm_away = self._normalize_name(away_name)

        # Search for the match
        for ev in data.get("events", []):
            if not ev.get("competitions"): continue
            comp = ev["competitions"][0]
            
            # Match teams
            teams = [c.get("team", {}).get("name", "") for c in comp.get("competitors", [])]
            if len(teams) < 2: continue
            
            t1, t2 = self._normalize_name(teams[0]), self._normalize_name(teams[1])
            
            # Basic fuzzy check
            match_found = False
            if (norm_home in t1 or t1 in norm_home or difflib.SequenceMatcher(None, norm_home, t1).ratio() > 0.7) and \
               (norm_away in t2 or t2 in norm_away or difflib.SequenceMatcher(None, norm_away, t2).ratio() > 0.7):
                match_found = True
            elif (norm_home in t2 or t2 in norm_home or difflib.SequenceMatcher(None, norm_home, t2).ratio() > 0.7) and \
                 (norm_away in t1 or t1 in norm_away or difflib.SequenceMatcher(None, norm_away, t1).ratio() > 0.7):
                match_found = True

            if match_found and "odds" in comp:
                odds_info = comp["odds"][0]
                provider = odds_info.get("provider", {}).get("name", "ESPN")
                
                # ESPN sometimes doesn't have 1X2 in standard format. We need home/away/draw
                # Some matches only have spread. Let's check for moneyline.
                if "moneyline" in odds_info:
                    ml = odds_info["moneyline"]
                    h_amer = ml.get("home", {}).get("current", {}).get("odds", "")
                    a_amer = ml.get("away", {}).get("current", {}).get("odds", "")
                    d_amer = ml.get("draw", {}).get("current", {}).get("odds", "")
                    
                    if h_amer and a_amer and d_amer:
                        h_dec = self._american_to_decimal(h_amer)
                        a_dec = self._american_to_decimal(a_amer)
                        d_dec = self._american_to_decimal(d_amer)
                        
                        if h_dec > 0 and a_dec > 0 and d_dec > 0:
                            imp_h = 1.0 / h_dec
                            imp_a = 1.0 / a_dec
                            imp_d = 1.0 / d_dec
                            
                            overround = imp_h + imp_d + imp_a
                            return {
                                "prob_home": (imp_h / overround) * 100.0,
                                "prob_draw": (imp_d / overround) * 100.0,
                                "prob_away": (imp_a / overround) * 100.0,
                                "bookmaker": provider,
                                "margin_removed": round((overround - 1.0) * 100, 2)
                            }
        return None

    def _fetch_odds(self, sport_key: str) -> list:
        if not self.enabled:
            return []
            
        cache_path = self._get_cache_path(sport_key)
        
        # Simple cache: odds update every 30 mins
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if datetime.now().timestamp() - mtime < 1800:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": "eu,uk",
                "markets": "h2h",
                "oddsFormat": "decimal"
            }
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            else:
                logger.warning(f"Failed to fetch odds for {sport_key}: {r.status_code} {r.text}")
                return []
        except Exception as e:
            logger.error(f"Error fetching odds for {sport_key}: {e}")
            return []

    def get_market_probabilities(self, league_code: str, home_name: str, away_name: str) -> dict | None:
        """
        Fetches the market odds for a match and returns the true implied probabilities
        (margin removed). Tries ESPN (free) first, falls back to The Odds API.
        """
        # 1. Try ESPN (Free, no API key needed)
        espn_code = ESPN_MAP.get(league_code)
        if espn_code:
            espn_probs = self._fetch_espn_odds(espn_code, home_name, away_name)
            if espn_probs:
                logger.info(f"[{league_code}] Found odds via ESPN ({espn_probs['bookmaker']})")
                return espn_probs

        # 2. Fallback to The Odds API
        if not self.enabled:
            return None
            
        sport_key = LEAGUE_MAP.get(league_code)
        if not sport_key:
            return None
            
        odds_data = self._fetch_odds(sport_key)
        if not odds_data:
            return None

        # Find the match
        # We need to fuzzy match team names since API providers use different naming conventions
        norm_target_home = self._normalize_name(home_name)
        norm_target_away = self._normalize_name(away_name)
        
        match = None
        for m in odds_data:
            h_api = self._normalize_name(m["home_team"])
            a_api = self._normalize_name(m["away_team"])
            
            # Direct match
            if (norm_target_home in h_api or h_api in norm_target_home) and (norm_target_away in a_api or a_api in norm_target_away):
                match = m
                break
                
            # Fuzzy match (difflib)
            if difflib.SequenceMatcher(None, norm_target_home, h_api).ratio() > 0.7 and \
               difflib.SequenceMatcher(None, norm_target_away, a_api).ratio() > 0.7:
                match = m
                break

        if not match:
            return None

        # Extract Pinnacle odds (usually the sharpest market) or fallback to any bookmaker
        bookies = match.get("bookmakers", [])
        if not bookies:
            return None
            
        # Try to find pinnacle, otherwise just take the first one (often bet365 or similar in eu)
        selected_bookie = next((b for b in bookies if b["key"] == "pinnacle"), bookies[0])
        
        markets = selected_bookie.get("markets", [])
        if not markets:
            return None
            
        h2h_market = next((m for m in markets if m["key"] == "h2h"), None)
        if not h2h_market:
            return None
            
        outcomes = h2h_market.get("outcomes", [])
        if len(outcomes) != 3:
            return None

        # Map odds. The Odds API provides names: home team, away team, "Draw"
        odds = {"home": 0, "draw": 0, "away": 0}
        for outcome in outcomes:
            name = outcome["name"]
            price = outcome["price"]
            if name == match["home_team"]:
                odds["home"] = price
            elif name == match["away_team"]:
                odds["away"] = price
            elif name == "Draw":
                odds["draw"] = price
                
        if not all(odds.values()):
            return None

        # Convert to implied probabilities
        imp_h = 1.0 / odds["home"]
        imp_d = 1.0 / odds["draw"]
        imp_a = 1.0 / odds["away"]
        
        # Calculate overround (margin)
        overround = imp_h + imp_d + imp_a
        
        # Remove margin proportionally to get true probabilities
        true_h = (imp_h / overround) * 100.0
        true_d = (imp_d / overround) * 100.0
        true_a = (imp_a / overround) * 100.0
        
        return {
            "prob_home": true_h,
            "prob_draw": true_d,
            "prob_away": true_a,
            "bookmaker": selected_bookie["title"],
            "margin_removed": round((overround - 1.0) * 100, 2)
        }

    def get_live_matches(self, league_code: str) -> dict:
        """
        Returns a dict of currently live or today's matches from ESPN.
        Format: {(norm_home, norm_away): {'state': 'in'/'pre'/'post', 'score': '1 - 0', 'minute': '52\\''}}
        """
        espn_code = ESPN_MAP.get(league_code)
        if not espn_code:
            return {}
            
        # Short cache (1 min) to avoid spamming ESPN
        cache_path = self._get_cache_path(espn_code, prefix="live_")
        data = None
        if os.path.exists(cache_path):
            if datetime.now().timestamp() - os.path.getmtime(cache_path) < 60:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    pass
        
        if not data:
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard"
                r = self.session.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
            except Exception as e:
                logger.error(f"Error fetching live scoreboard for {league_code}: {e}")
                return {}
                
        if not data: return {}
        
        live_dict = {}
        for ev in data.get("events", []):
            comp = ev.get("competitions", [])[0] if ev.get("competitions") else {}
            competitors = comp.get("competitors", [])
            if len(competitors) < 2: continue
            
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            
            h_name = self._normalize_name(home_c.get("team", {}).get("name", ""))
            a_name = self._normalize_name(away_c.get("team", {}).get("name", ""))
            h_score = home_c.get("score", "0")
            a_score = away_c.get("score", "0")
            
            status = ev.get("status", {}).get("type", {})
            state = status.get("state", "pre")  # 'in', 'pre', 'post'
            minute = status.get("shortDetail", "")
            
            live_dict[(h_name, a_name)] = {
                "state": state,
                "score": f"{h_score} - {a_score}",
                "minute": minute
            }
            
        return live_dict

market_service = MarketService()
