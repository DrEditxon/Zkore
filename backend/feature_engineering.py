import pandas as pd
import numpy as np

def _init_team_state():
    return {
        "elo": 1500.0,
        "points": 0,
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

def _update_elo(rating_a, rating_b, actual_score_a, k=20):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return rating_a + k * (actual_score_a - expected_a)

def _ewm_mean(values, alpha=0.5):
    if not values: return 0.0
    series = pd.Series(values)
    return float(series.ewm(alpha=alpha, adjust=False).mean().iloc[-1])

def _compute_all_features(matches: list) -> tuple:
    if not matches:
        return [], {}, {}
    
    df = pd.DataFrame(matches)
    df["utcDate"] = pd.to_datetime(df["utcDate"])
    df = df.sort_values("utcDate").reset_index(drop=True)
    
    team_states = {}
    h2h_states = {} # keys: (teamA, teamB) ascending. vals: {w_A, w_B, g_A, g_B}
    features_list = []
    
    for idx, match in df.iterrows():
        home_id = match["homeTeam_id"]
        away_id = match["awayTeam_id"]
        home_goals = match["homeGoals"]
        away_goals = match["awayGoals"]
        match_date = match["utcDate"]
        
        if home_id not in team_states: team_states[home_id] = _init_team_state()
        if away_id not in team_states: team_states[away_id] = _init_team_state()
        
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
            
            if home_goals > away_goals: result = 2
            elif home_goals == away_goals: result = 1
            else: result = 0
            
            row = {
                "home_elo": h_state["elo"],
                "away_elo": a_state["elo"],
                "elo_diff": h_state["elo"] - a_state["elo"],
                
                "home_points": h_state["points"],
                "away_points": a_state["points"],
                "points_diff": h_state["points"] - a_state["points"],
                
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
                
                "home_gs_ewm": _ewm_mean(h_state["recent_goals_scored"]),
                "home_gc_ewm": _ewm_mean(h_state["recent_goals_conceded"]),
                "away_gs_ewm": _ewm_mean(a_state["recent_goals_scored"]),
                "away_gc_ewm": _ewm_mean(a_state["recent_goals_conceded"]),
                
                "home_cs_rate": h_state["clean_sheets"] / h_mp,
                "away_cs_rate": a_state["clean_sheets"] / a_mp,
                "home_fts_rate": h_state["failed_to_score"] / h_mp,
                "away_fts_rate": a_state["failed_to_score"] / a_mp,
                
                "h2h_home_wins": h2h_home_wins,
                "h2h_away_wins": h2h_away_wins,
                "h2h_gd": h2h_home_goals - h2h_away_goals,
                
                "result": result,
                "home_goals_cat": min(3, int(home_goals)),
                "away_goals_cat": min(3, int(away_goals))
            }
            features_list.append(row)
            
        # UPDATE STATE POST-MATCH
        h_actual = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)
        a_actual = 1.0 - h_actual
        new_h_elo = _update_elo(h_state["elo"], a_state["elo"], h_actual)
        new_a_elo = _update_elo(a_state["elo"], h_state["elo"], a_actual)
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

def build_training_dataframe(matches: list) -> pd.DataFrame:
    features_list, _, _ = _compute_all_features(matches)
    valid_rows = [r for r in features_list if r is not None]
    return pd.DataFrame(valid_rows)

def build_prediction_features(home_id: int, away_id: int, matches: list) -> pd.DataFrame:
    _, team_states, h2h_states = _compute_all_features(matches)
    
    h_state = team_states.get(home_id, _init_team_state())
    a_state = team_states.get(away_id, _init_team_state())
    
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
    
    row = {
        "home_elo": h_state["elo"],
        "away_elo": a_state["elo"],
        "elo_diff": h_state["elo"] - a_state["elo"],
        
        "home_points": h_state["points"],
        "away_points": a_state["points"],
        "points_diff": h_state["points"] - a_state["points"],
        
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
        
        "home_gs_ewm": _ewm_mean(h_state["recent_goals_scored"]),
        "home_gc_ewm": _ewm_mean(h_state["recent_goals_conceded"]),
        "away_gs_ewm": _ewm_mean(a_state["recent_goals_scored"]),
        "away_gc_ewm": _ewm_mean(a_state["recent_goals_conceded"]),
        
        "home_cs_rate": h_state["clean_sheets"] / h_mp,
        "away_cs_rate": a_state["clean_sheets"] / a_mp,
        "home_fts_rate": h_state["failed_to_score"] / h_mp,
        "away_fts_rate": a_state["failed_to_score"] / a_mp,
        
        "h2h_home_wins": h2h["w_A"] if is_home_A else h2h["w_B"],
        "h2h_away_wins": h2h["w_B"] if is_home_A else h2h["w_A"],
        "h2h_gd": (h2h["g_A"] - h2h["g_B"]) if is_home_A else (h2h["g_B"] - h2h["g_A"]),
    }
    
    return pd.DataFrame([row])
