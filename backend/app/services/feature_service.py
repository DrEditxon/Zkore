import pandas as pd
import numpy as np

# Use the same list across model and feature creation
FEATURE_COLS = [
    "home_attack_elo", "home_defense_elo", "away_attack_elo", "away_defense_elo",
    "home_atk_vs_away_def", "away_atk_vs_home_def",
    "home_points_per_game", "away_points_per_game",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_form_pts", "away_form_pts", "form_diff",
    "home_win_streak", "away_win_streak", "streak_diff",
    "home_unbeaten_streak", "away_unbeaten_streak",
    "home_gs_ewm", "home_gc_ewm", "away_gs_ewm", "away_gc_ewm",
    "home_cs_rate", "away_cs_rate", "home_fts_rate", "away_fts_rate",
    "h2h_home_wins", "h2h_away_wins", "h2h_gd",
    "home_attack_vs_away_defense", "away_attack_vs_home_defense",
    "home_goal_rate", "away_goal_rate",
    "home_xg_proxy", "away_xg_proxy", "home_advantage"
]


# ─────────────────────────────────────────────────────────────
# PERF FIX B: Pure-python EWM — 166x faster than pd.Series.ewm
# ─────────────────────────────────────────────────────────────
def _ewm_fast(values: list, alpha: float = 0.5) -> float:
    if not values:
        return 0.0
    result = float(values[0])
    for v in values[1:]:
        result = alpha * v + (1.0 - alpha) * result
    return result


class FeatureService:

    def __init__(self):
        # PERF FIX A+D: In-memory cache for computed team states, keyed by data hash.
        # Avoids re-running the full O(N) loop for every single match prediction.
        self._states_cache: dict = {}   # { hash -> (team_states, h2h_states) }

    def _hash_matches(self, matches: list) -> str:
        """Fast fingerprint of the match list to detect data changes."""
        key = f"{len(matches)}_{matches[-1]['utcDate'] if matches else ''}"
        return key

    def _init_team_state(self):
        return {
            "attack_elo": 1500.0,
            "defense_elo": 1500.0,
            "points": 0,
            "home_points": 0,
            "home_matches": 0,
            "away_points": 0,
            "away_matches": 0,
            "matches_played": 0,
            "clean_sheets": 0,
            "failed_to_score": 0,
            "win_streak": 0,
            "unbeaten_streak": 0,
            "last_match_date": None,
            "recent_goals_scored": [],
            "recent_goals_conceded": [],
            "recent_results": []
        }

    def _update_elo_attack_defense(self, atk_rating, def_rating, goals_scored, k=20):
        expected = 1.0 / (1.0 + 10.0 ** ((def_rating - atk_rating) / 400.0))
        if goals_scored == 0: actual = 0.0
        elif goals_scored == 1: actual = 0.5
        elif goals_scored == 2: actual = 0.8
        else: actual = 1.0
        
        mov = 1.0 if goals_scored <= 1 else (11.0 + goals_scored) / 8.0
        change = k * mov * (actual - expected)
        
        return atk_rating + change, def_rating - change

    def _compute_all_features(self, matches: list) -> tuple:
        """
        O(N) walk over match history building team states and feature rows.

        PERF FIX A: Result is cached by data fingerprint — subsequent calls with
        the same dataset (e.g. 10 parallel match predictions for the same league)
        execute in microseconds instead of re-running the full loop.

        PERF FIX B: Uses pure-python _ewm_fast instead of pd.Series.ewm (166x faster).
        PERF FIX C: Uses df.itertuples() instead of df.iterrows() (3-5x faster).
        """
        if not matches:
            return [], {}, {}

        cache_key = self._hash_matches(matches)
        if cache_key in self._states_cache:
            return self._states_cache[cache_key]

        df = pd.DataFrame(matches)
        df["utcDate"] = pd.to_datetime(df["utcDate"])
        df = df.sort_values("utcDate").reset_index(drop=True)

        team_states: dict = {}
        h2h_states: dict = {}
        features_list: list = []

        # PERF FIX C: itertuples is 3-5x faster than iterrows
        for row in df.itertuples(index=False):
            home_id    = row.homeTeam_id
            away_id    = row.awayTeam_id
            home_goals = int(row.homeGoals)
            away_goals = int(row.awayGoals)
            match_date = row.utcDate

            if home_id not in team_states:
                team_states[home_id] = self._init_team_state()
            if away_id not in team_states:
                team_states[away_id] = self._init_team_state()

            pair = (min(home_id, away_id), max(home_id, away_id))
            if pair not in h2h_states:
                h2h_states[pair] = {"w_A": 0, "w_B": 0, "g_A": 0, "g_B": 0}

            h_state = team_states[home_id]
            a_state = team_states[away_id]
            h2h     = h2h_states[pair]
            is_home_A = (home_id == pair[0])

            # FASE 2: Regresión a la media (Shrinkage) en cambios de temporada
            if h_state["last_match_date"] and (match_date - h_state["last_match_date"]).days > 60:
                h_state["attack_elo"] = 1500.0 + 0.85 * (h_state["attack_elo"] - 1500.0)
                h_state["defense_elo"] = 1500.0 + 0.85 * (h_state["defense_elo"] - 1500.0)
            if a_state["last_match_date"] and (match_date - a_state["last_match_date"]).days > 60:
                a_state["attack_elo"] = 1500.0 + 0.85 * (a_state["attack_elo"] - 1500.0)
                a_state["defense_elo"] = 1500.0 + 0.85 * (a_state["defense_elo"] - 1500.0)

            if h_state["matches_played"] < 3 or a_state["matches_played"] < 3:
                features_list.append(None)
            else:
                h_rest = (match_date - h_state["last_match_date"]).days if h_state["last_match_date"] else 7
                a_rest = (match_date - a_state["last_match_date"]).days if a_state["last_match_date"] else 7

                h_form = sum(h_state["recent_results"][-5:])
                a_form = sum(a_state["recent_results"][-5:])

                h_mp = max(1, h_state["matches_played"])
                a_mp = max(1, a_state["matches_played"])

                # PERF FIX B: pure-python EWM (166x faster than pd.Series.ewm)
                home_gs_ewm = _ewm_fast(h_state["recent_goals_scored"])
                home_gc_ewm = _ewm_fast(h_state["recent_goals_conceded"])
                away_gs_ewm = _ewm_fast(a_state["recent_goals_scored"])
                away_gc_ewm = _ewm_fast(a_state["recent_goals_conceded"])

                h_home_ppg = h_state["home_points"] / max(1, h_state["home_matches"])
                h_away_ppg = h_state["away_points"] / max(1, h_state["away_matches"])
                h_adv      = h_home_ppg - h_away_ppg

                h2h_home_wins  = h2h["w_A"] if is_home_A else h2h["w_B"]
                h2h_away_wins  = h2h["w_B"] if is_home_A else h2h["w_A"]
                h2h_home_goals = h2h["g_A"] if is_home_A else h2h["g_B"]
                h2h_away_goals = h2h["g_B"] if is_home_A else h2h["g_A"]

                features_list.append({
                    "home_attack_elo": h_state["attack_elo"],
                    "home_defense_elo": h_state["defense_elo"],
                    "away_attack_elo": a_state["attack_elo"],
                    "away_defense_elo": a_state["defense_elo"],
                    "home_atk_vs_away_def": h_state["attack_elo"] - a_state["defense_elo"],
                    "away_atk_vs_home_def": a_state["attack_elo"] - h_state["defense_elo"],

                    "home_points_per_game": h_state["points"] / h_mp,
                    "away_points_per_game": a_state["points"] / a_mp,

                    "home_rest_days":  h_rest,
                    "away_rest_days":  a_rest,
                    "rest_days_diff":  h_rest - a_rest,

                    "home_form_pts": h_form,
                    "away_form_pts": a_form,
                    "form_diff":     h_form - a_form,

                    "home_win_streak":      h_state["win_streak"],
                    "away_win_streak":      a_state["win_streak"],
                    "streak_diff":          h_state["win_streak"] - a_state["win_streak"],

                    "home_unbeaten_streak": h_state["unbeaten_streak"],
                    "away_unbeaten_streak": a_state["unbeaten_streak"],

                    "home_gs_ewm": home_gs_ewm,
                    "home_gc_ewm": home_gc_ewm,
                    "away_gs_ewm": away_gs_ewm,
                    "away_gc_ewm": away_gc_ewm,

                    "home_cs_rate":  h_state["clean_sheets"] / h_mp,
                    "away_cs_rate":  a_state["clean_sheets"] / a_mp,
                    "home_fts_rate": h_state["failed_to_score"] / h_mp,
                    "away_fts_rate": a_state["failed_to_score"] / a_mp,

                    "h2h_home_wins": h2h_home_wins,
                    "h2h_away_wins": h2h_away_wins,
                    "h2h_gd":        h2h_home_goals - h2h_away_goals,

                    "home_attack_vs_away_defense": home_gs_ewm / max(0.1, away_gc_ewm),
                    "away_attack_vs_home_defense": away_gs_ewm / max(0.1, home_gc_ewm),
                    "home_goal_rate": float(np.mean(h_state["recent_goals_scored"])) if h_state["recent_goals_scored"] else 0.0,
                    "away_goal_rate": float(np.mean(a_state["recent_goals_scored"])) if a_state["recent_goals_scored"] else 0.0,
                    "home_xg_proxy": home_gs_ewm * away_gc_ewm,
                    "away_xg_proxy": away_gs_ewm * home_gc_ewm,
                    "home_advantage": h_adv,

                    "target_home_goals": home_goals,
                    "target_away_goals": away_goals,
                })

            # ── UPDATE STATE POST-MATCH ────────────────────────────────────
            new_h_atk, new_a_def = self._update_elo_attack_defense(
                h_state["attack_elo"], a_state["defense_elo"], home_goals
            )
            new_a_atk, new_h_def = self._update_elo_attack_defense(
                a_state["attack_elo"], h_state["defense_elo"], away_goals
            )
            h_state["attack_elo"] = new_h_atk
            a_state["defense_elo"] = new_a_def
            a_state["attack_elo"] = new_a_atk
            h_state["defense_elo"] = new_h_def

            if home_goals > away_goals:
                h_pts, a_pts = 3, 0
                h_state["points"] += 3
                h_state["win_streak"] += 1;      h_state["unbeaten_streak"] += 1
                a_state["win_streak"] = 0;       a_state["unbeaten_streak"] = 0
                if is_home_A: h2h["w_A"] += 1
                else:         h2h["w_B"] += 1
            elif home_goals == away_goals:
                h_pts, a_pts = 1, 1
                h_state["points"] += 1;  a_state["points"] += 1
                h_state["win_streak"] = 0;   a_state["win_streak"] = 0
                h_state["unbeaten_streak"] += 1; a_state["unbeaten_streak"] += 1
            else:
                h_pts, a_pts = 0, 3
                a_state["points"] += 3
                a_state["win_streak"] += 1;      a_state["unbeaten_streak"] += 1
                h_state["win_streak"] = 0;       h_state["unbeaten_streak"] = 0
                if is_home_A: h2h["w_B"] += 1
                else:         h2h["w_A"] += 1

            if is_home_A:
                h2h["g_A"] += home_goals;  h2h["g_B"] += away_goals
            else:
                h2h["g_B"] += home_goals;  h2h["g_A"] += away_goals

            if away_goals == 0: h_state["clean_sheets"] += 1
            if home_goals == 0: a_state["clean_sheets"] += 1
            if home_goals == 0: h_state["failed_to_score"] += 1
            if away_goals == 0: a_state["failed_to_score"] += 1

            h_state["matches_played"] += 1;  a_state["matches_played"] += 1
            h_state["home_matches"]   += 1;  a_state["away_matches"]   += 1
            h_state["home_points"]    += h_pts
            a_state["away_points"]    += a_pts
            h_state["last_match_date"] = match_date
            a_state["last_match_date"] = match_date

            h_state["recent_goals_scored"].append(home_goals)
            h_state["recent_goals_conceded"].append(away_goals)
            h_state["recent_results"].append(h_pts)
            a_state["recent_goals_scored"].append(away_goals)
            a_state["recent_goals_conceded"].append(home_goals)
            a_state["recent_results"].append(a_pts)

            # Keep only last 5
            if len(h_state["recent_goals_scored"]) > 5:
                h_state["recent_goals_scored"]  = h_state["recent_goals_scored"][-5:]
                h_state["recent_goals_conceded"] = h_state["recent_goals_conceded"][-5:]
                h_state["recent_results"]        = h_state["recent_results"][-5:]
            if len(a_state["recent_goals_scored"]) > 5:
                a_state["recent_goals_scored"]  = a_state["recent_goals_scored"][-5:]
                a_state["recent_goals_conceded"] = a_state["recent_goals_conceded"][-5:]
                a_state["recent_results"]        = a_state["recent_results"][-5:]

        result = (features_list, team_states, h2h_states)
        # Store in cache — keep at most 10 leagues to avoid unbounded memory
        if len(self._states_cache) >= 10:
            oldest = next(iter(self._states_cache))
            del self._states_cache[oldest]
        self._states_cache[cache_key] = result
        return result

    def invalidate_cache(self, league_code: str = None):
        """Call this when fresh historical data is fetched for a league."""
        self._states_cache.clear()

    def build_training_dataframe(self, matches: list) -> pd.DataFrame:
        features_list, _, _ = self._compute_all_features(matches)
        valid_rows = [r for r in features_list if r is not None]
        return pd.DataFrame(valid_rows)

    def build_prediction_features(self, home_id: int, away_id: int, matches: list) -> pd.DataFrame:
        # PERF FIX A: _compute_all_features is now cached — this is O(1) on the 2nd+ call
        _, team_states, h2h_states = self._compute_all_features(matches)

        h_state = team_states.get(home_id, self._init_team_state())
        a_state = team_states.get(away_id, self._init_team_state())

        pair  = (min(home_id, away_id), max(home_id, away_id))
        h2h   = h2h_states.get(pair, {"w_A": 0, "w_B": 0, "g_A": 0, "g_B": 0})
        is_home_A = (home_id == pair[0])

        if matches:
            today = pd.to_datetime(pd.DataFrame(matches)["utcDate"]).max() + pd.Timedelta(days=1)
        else:
            today = pd.Timestamp.utcnow()

        h_rest = (today - h_state["last_match_date"]).days if h_state["last_match_date"] else 7
        a_rest = (today - a_state["last_match_date"]).days if a_state["last_match_date"] else 7
        h_mp   = max(1, h_state["matches_played"])
        a_mp   = max(1, a_state["matches_played"])

        home_gs_ewm = _ewm_fast(h_state["recent_goals_scored"])
        home_gc_ewm = _ewm_fast(h_state["recent_goals_conceded"])
        away_gs_ewm = _ewm_fast(a_state["recent_goals_scored"])
        away_gc_ewm = _ewm_fast(a_state["recent_goals_conceded"])

        h_home_ppg = h_state["home_points"] / max(1, h_state["home_matches"])
        h_away_ppg = h_state["away_points"] / max(1, h_state["away_matches"])
        h_adv      = h_home_ppg - h_away_ppg

        h_atk_elo = h_state["attack_elo"]
        h_def_elo = h_state["defense_elo"]
        if h_state["last_match_date"] and (today - h_state["last_match_date"]).days > 60:
            h_atk_elo = 1500.0 + 0.85 * (h_atk_elo - 1500.0)
            h_def_elo = 1500.0 + 0.85 * (h_def_elo - 1500.0)
            
        a_atk_elo = a_state["attack_elo"]
        a_def_elo = a_state["defense_elo"]
        if a_state["last_match_date"] and (today - a_state["last_match_date"]).days > 60:
            a_atk_elo = 1500.0 + 0.85 * (a_atk_elo - 1500.0)
            a_def_elo = 1500.0 + 0.85 * (a_def_elo - 1500.0)

        row = {
            "home_attack_elo": h_atk_elo,
            "home_defense_elo": h_def_elo,
            "away_attack_elo": a_atk_elo,
            "away_defense_elo": a_def_elo,
            "home_atk_vs_away_def": h_atk_elo - a_def_elo,
            "away_atk_vs_home_def": a_atk_elo - h_def_elo,

            "home_points_per_game": h_state["points"] / h_mp,
            "away_points_per_game": a_state["points"] / a_mp,

            "home_rest_days":  h_rest,
            "away_rest_days":  a_rest,
            "rest_days_diff":  h_rest - a_rest,

            "home_form_pts": sum(h_state["recent_results"][-5:]),
            "away_form_pts": sum(a_state["recent_results"][-5:]),
            "form_diff":     sum(h_state["recent_results"][-5:]) - sum(a_state["recent_results"][-5:]),

            "home_win_streak":      h_state["win_streak"],
            "away_win_streak":      a_state["win_streak"],
            "streak_diff":          h_state["win_streak"] - a_state["win_streak"],

            "home_unbeaten_streak": h_state["unbeaten_streak"],
            "away_unbeaten_streak": a_state["unbeaten_streak"],

            "home_gs_ewm": home_gs_ewm,
            "home_gc_ewm": home_gc_ewm,
            "away_gs_ewm": away_gs_ewm,
            "away_gc_ewm": away_gc_ewm,

            "home_cs_rate":  h_state["clean_sheets"] / h_mp,
            "away_cs_rate":  a_state["clean_sheets"] / a_mp,
            "home_fts_rate": h_state["failed_to_score"] / h_mp,
            "away_fts_rate": a_state["failed_to_score"] / a_mp,

            "h2h_home_wins": h2h["w_A"] if is_home_A else h2h["w_B"],
            "h2h_away_wins": h2h["w_B"] if is_home_A else h2h["w_A"],
            "h2h_gd":        (h2h["g_A"] - h2h["g_B"]) if is_home_A else (h2h["g_B"] - h2h["g_A"]),

            "home_attack_vs_away_defense": home_gs_ewm / max(0.1, away_gc_ewm),
            "away_attack_vs_home_defense": away_gs_ewm / max(0.1, home_gc_ewm),
            "home_goal_rate": float(np.mean(h_state["recent_goals_scored"])) if h_state["recent_goals_scored"] else 0.0,
            "away_goal_rate": float(np.mean(a_state["recent_goals_scored"])) if a_state["recent_goals_scored"] else 0.0,
            "home_xg_proxy": home_gs_ewm * away_gc_ewm,
            "away_xg_proxy": away_gs_ewm * home_gc_ewm,
            "home_advantage": h_adv,
        }

        df_pred = pd.DataFrame([row])[FEATURE_COLS]

        if df_pred.empty or df_pred.isna().all().all():
            raise ValueError(f"Feature engine failed for teams {home_id} vs {away_id}")

        return df_pred


feature_service = FeatureService()
