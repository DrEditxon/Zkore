import pandas as pd
import numpy as np

# Use the same list across model and feature creation
FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
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

class FeatureService:

    def _init_team_state(self):
        return {
            "elo": 1500.0,
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

    def _update_elo(self, rating_a, rating_b, actual_score_a, margin, k=20):
        """
        Improved ELO that factors in margin of victory.
        margin = abs(goals_A - goals_B)
        """
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        # Margin of victory multiplier
        if margin == 0 or margin == 1:
            mov_multiplier = 1.0
        elif margin == 2:
            mov_multiplier = 1.5
        else:
            mov_multiplier = (11 + margin) / 8.0
            
        return rating_a + k * mov_multiplier * (actual_score_a - expected_a)

    def _ewm_mean(self, values, alpha=0.5):
        if not values: return 0.0
        series = pd.Series(values)
        return float(series.ewm(alpha=alpha, adjust=False).mean().iloc[-1])

    def _compute_all_features(self, matches: list) -> tuple:
        if not matches:
            return [], {}, {}
        
        df = pd.DataFrame(matches)
        df["utcDate"] = pd.to_datetime(df["utcDate"])
        df = df.sort_values("utcDate").reset_index(drop=True)
        
        team_states = {}
        h2h_states = {}
        features_list = []
        
        for idx, match in df.iterrows():
            home_id = match["homeTeam_id"]
            away_id = match["awayTeam_id"]
            home_goals = match["homeGoals"]
            away_goals = match["awayGoals"]
            match_date = match["utcDate"]
            
            if home_id not in team_states: team_states[home_id] = self._init_team_state()
            if away_id not in team_states: team_states[away_id] = self._init_team_state()
            
            pair = tuple(sorted((home_id, away_id)))
            if pair not in h2h_states:
                h2h_states[pair] = {"w_A": 0, "w_B": 0, "g_A": 0, "g_B": 0}
                
            h_state = team_states[home_id]
            a_state = team_states[away_id]
            h2h = h2h_states[pair]
            
            is_home_A = (home_id == pair[0])
            
            if h_state["matches_played"] < 3 or a_state["matches_played"] < 3:
                features_list.append(None)
            else:
                h_rest = (match_date - h_state["last_match_date"]).days if h_state["last_match_date"] else 7
                a_rest = (match_date - a_state["last_match_date"]).days if a_state["last_match_date"] else 7
                
                h_form = sum(h_state["recent_results"][-5:])
                a_form = sum(a_state["recent_results"][-5:])
                
                h_mp = max(1, h_state["matches_played"])
                a_mp = max(1, a_state["matches_played"])
                
                h2h_home_wins = h2h["w_A"] if is_home_A else h2h["w_B"]
                h2h_away_wins = h2h["w_B"] if is_home_A else h2h["w_A"]
                h2h_home_goals = h2h["g_A"] if is_home_A else h2h["g_B"]
                h2h_away_goals = h2h["g_B"] if is_home_A else h2h["g_A"]
                
                home_gs_ewm = self._ewm_mean(h_state["recent_goals_scored"])
                home_gc_ewm = self._ewm_mean(h_state["recent_goals_conceded"])
                away_gs_ewm = self._ewm_mean(a_state["recent_goals_scored"])
                away_gc_ewm = self._ewm_mean(a_state["recent_goals_conceded"])
                
                h_home_ppg = h_state["home_points"] / max(1, h_state["home_matches"])
                h_away_ppg = h_state["away_points"] / max(1, h_state["away_matches"])
                h_adv = h_home_ppg - h_away_ppg
                
                row = {
                    "home_elo": h_state["elo"],
                    "away_elo": a_state["elo"],
                    "elo_diff": h_state["elo"] - a_state["elo"],
                    
                    "home_points_per_game": h_state["points"] / h_mp,
                    "away_points_per_game": a_state["points"] / a_mp,
                    
                    "home_rest_days": h_rest,
                    "away_rest_days": a_rest,
                    "rest_days_diff": h_rest - a_rest,
                    
                    "home_form_pts": h_form,
                    "away_form_pts": a_form,
                    "form_diff": h_form - a_form,
                    
                    "home_win_streak": h_state["win_streak"],
                    "away_win_streak": a_state["win_streak"],
                    "streak_diff": h_state["win_streak"] - a_state["win_streak"],
                    
                    "home_unbeaten_streak": h_state["unbeaten_streak"],
                    "away_unbeaten_streak": a_state["unbeaten_streak"],
                    
                    "home_gs_ewm": home_gs_ewm,
                    "home_gc_ewm": home_gc_ewm,
                    "away_gs_ewm": away_gs_ewm,
                    "away_gc_ewm": away_gc_ewm,
                    
                    "home_cs_rate": h_state["clean_sheets"] / h_mp,
                    "away_cs_rate": a_state["clean_sheets"] / a_mp,
                    "home_fts_rate": h_state["failed_to_score"] / h_mp,
                    "away_fts_rate": a_state["failed_to_score"] / a_mp,
                    
                    "h2h_home_wins": h2h_home_wins,
                    "h2h_away_wins": h2h_away_wins,
                    "h2h_gd": h2h_home_goals - h2h_away_goals,
                    
                    # New metrics
                    "home_attack_vs_away_defense": home_gs_ewm / max(0.1, away_gc_ewm),
                    "away_attack_vs_home_defense": away_gs_ewm / max(0.1, home_gc_ewm),
                    "home_goal_rate": np.mean(h_state["recent_goals_scored"]) if h_state["recent_goals_scored"] else 0,
                    "away_goal_rate": np.mean(a_state["recent_goals_scored"]) if a_state["recent_goals_scored"] else 0,
                    "home_xg_proxy": home_gs_ewm * away_gc_ewm,
                    "away_xg_proxy": away_gs_ewm * home_gc_ewm,
                    "home_advantage": h_adv,
                    
                    # Labels (Target variables for Regressor)
                    "target_home_goals": home_goals,
                    "target_away_goals": away_goals
                }
                features_list.append(row)
                
            # UPDATE STATE POST-MATCH
            margin = abs(home_goals - away_goals)
            h_actual = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
            a_actual = 1.0 - h_actual
            new_h_elo = self._update_elo(h_state["elo"], a_state["elo"], h_actual, margin)
            new_a_elo = self._update_elo(a_state["elo"], h_state["elo"], a_actual, margin)
            h_state["elo"] = new_h_elo
            a_state["elo"] = new_a_elo
            
            if home_goals > away_goals:
                h_state["points"] += 3
                h_state["win_streak"] += 1
                h_state["unbeaten_streak"] += 1
                a_state["win_streak"] = 0
                a_state["unbeaten_streak"] = 0
                h_pts, a_pts = 3, 0
                if is_home_A: h2h["w_A"] += 1 
                else: h2h["w_B"] += 1
            elif home_goals == away_goals:
                h_state["points"] += 1
                a_state["points"] += 1
                h_state["win_streak"] = 0
                a_state["win_streak"] = 0
                h_state["unbeaten_streak"] += 1
                a_state["unbeaten_streak"] += 1
                h_pts, a_pts = 1, 1
            else:
                a_state["points"] += 3
                a_state["win_streak"] += 1
                a_state["unbeaten_streak"] += 1
                h_state["win_streak"] = 0
                h_state["unbeaten_streak"] = 0
                h_pts, a_pts = 0, 3
                if is_home_A: h2h["w_B"] += 1 
                else: h2h["w_A"] += 1
                
            if is_home_A:
                h2h["g_A"] += home_goals
                h2h["g_B"] += away_goals
            else:
                h2h["g_B"] += home_goals
                h2h["g_A"] += away_goals
                
            if away_goals == 0: h_state["clean_sheets"] += 1
            if home_goals == 0: a_state["clean_sheets"] += 1
            if home_goals == 0: h_state["failed_to_score"] += 1
            if away_goals == 0: a_state["failed_to_score"] += 1
            
            h_state["matches_played"] += 1
            a_state["matches_played"] += 1
            
            h_state["home_matches"] += 1
            a_state["away_matches"] += 1
            
            h_state["home_points"] += h_pts
            a_state["away_points"] += a_pts
            
            h_state["last_match_date"] = match_date
            a_state["last_match_date"] = match_date
            
            h_state["recent_goals_scored"].append(home_goals)
            h_state["recent_goals_conceded"].append(away_goals)
            h_state["recent_results"].append(h_pts)
            
            a_state["recent_goals_scored"].append(away_goals)
            a_state["recent_goals_conceded"].append(home_goals)
            a_state["recent_results"].append(a_pts)

            h_state["recent_goals_scored"] = h_state["recent_goals_scored"][-5:]
            h_state["recent_goals_conceded"] = h_state["recent_goals_conceded"][-5:]
            h_state["recent_results"] = h_state["recent_results"][-5:]
            
            a_state["recent_goals_scored"] = a_state["recent_goals_scored"][-5:]
            a_state["recent_goals_conceded"] = a_state["recent_goals_conceded"][-5:]
            a_state["recent_results"] = a_state["recent_results"][-5:]

        return features_list, team_states, h2h_states

    def build_training_dataframe(self, matches: list) -> pd.DataFrame:
        features_list, _, _ = self._compute_all_features(matches)
        valid_rows = [r for r in features_list if r is not None]
        return pd.DataFrame(valid_rows)

    def build_prediction_features(self, home_id: int, away_id: int, matches: list) -> pd.DataFrame:
        _, team_states, h2h_states = self._compute_all_features(matches)
        
        h_state = team_states.get(home_id, self._init_team_state())
        a_state = team_states.get(away_id, self._init_team_state())
        
        pair = tuple(sorted((home_id, away_id)))
        h2h = h2h_states.get(pair, {"w_A": 0, "w_B": 0, "g_A": 0, "g_B": 0})
        is_home_A = (home_id == pair[0])
        
        if len(matches) > 0 and "utcDate" in pd.DataFrame(matches).columns:
            today = pd.to_datetime(pd.DataFrame(matches)["utcDate"]).max() + pd.Timedelta(days=1)
        else:
            today = pd.Timestamp.utcnow()
            
        h_rest = (today - h_state["last_match_date"]).days if h_state["last_match_date"] else 7
        a_rest = (today - a_state["last_match_date"]).days if a_state["last_match_date"] else 7
        
        h_mp = max(1, h_state["matches_played"])
        a_mp = max(1, a_state["matches_played"])
        
        home_gs_ewm = self._ewm_mean(h_state["recent_goals_scored"])
        home_gc_ewm = self._ewm_mean(h_state["recent_goals_conceded"])
        away_gs_ewm = self._ewm_mean(a_state["recent_goals_scored"])
        away_gc_ewm = self._ewm_mean(a_state["recent_goals_conceded"])

        h_home_ppg = h_state["home_points"] / max(1, h_state["home_matches"])
        h_away_ppg = h_state["away_points"] / max(1, h_state["away_matches"])
        h_adv = h_home_ppg - h_away_ppg

        row = {
            "home_elo": h_state["elo"],
            "away_elo": a_state["elo"],
            "elo_diff": h_state["elo"] - a_state["elo"],
            
            "home_points_per_game": h_state["points"] / h_mp,
            "away_points_per_game": a_state["points"] / a_mp,
            
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "rest_days_diff": h_rest - a_rest,
            
            "home_form_pts": sum(h_state["recent_results"][-5:]),
            "away_form_pts": sum(a_state["recent_results"][-5:]),
            "form_diff": sum(h_state["recent_results"][-5:]) - sum(a_state["recent_results"][-5:]),
            
            "home_win_streak": h_state["win_streak"],
            "away_win_streak": a_state["win_streak"],
            "streak_diff": h_state["win_streak"] - a_state["win_streak"],
            
            "home_unbeaten_streak": h_state["unbeaten_streak"],
            "away_unbeaten_streak": a_state["unbeaten_streak"],
            
            "home_gs_ewm": home_gs_ewm,
            "home_gc_ewm": home_gc_ewm,
            "away_gs_ewm": away_gs_ewm,
            "away_gc_ewm": away_gc_ewm,
            
            "home_cs_rate": h_state["clean_sheets"] / h_mp,
            "away_cs_rate": a_state["clean_sheets"] / a_mp,
            "home_fts_rate": h_state["failed_to_score"] / h_mp,
            "away_fts_rate": a_state["failed_to_score"] / a_mp,
            
            "h2h_home_wins": h2h["w_A"] if is_home_A else h2h["w_B"],
            "h2h_away_wins": h2h["w_B"] if is_home_A else h2h["w_A"],
            "h2h_gd": (h2h["g_A"] - h2h["g_B"]) if is_home_A else (h2h["g_B"] - h2h["g_A"]),

            # New metrics
            "home_attack_vs_away_defense": home_gs_ewm / max(0.1, away_gc_ewm),
            "away_attack_vs_home_defense": away_gs_ewm / max(0.1, home_gc_ewm),
            "home_goal_rate": np.mean(h_state["recent_goals_scored"]) if h_state["recent_goals_scored"] else 0,
            "away_goal_rate": np.mean(a_state["recent_goals_scored"]) if a_state["recent_goals_scored"] else 0,
            "home_xg_proxy": home_gs_ewm * away_gc_ewm,
            "away_xg_proxy": away_gs_ewm * home_gc_ewm,
            "home_advantage": h_adv,
        }
        
        df_pred = pd.DataFrame([row])
        
        df_pred = df_pred[FEATURE_COLS] # re-order strictly
        
        if df_pred.empty or df_pred.isna().all().all():
            raise ValueError(f"Feature engine failed generating features for teams {home_id} vs {away_id}")
            
        return df_pred

feature_service = FeatureService()
