import os
import pickle
import logging

import numpy as np
from scipy.stats import poisson
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split

from .data_fetcher import get_historical_matches
from .feature_engineering import build_training_dataframe, build_prediction_features

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MIN_MATCHES_REQUIRED = 30
MAX_GOALS = 5          # Max goals considered per team for scoreline matrix

_model_cache: dict = {}

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_points", "away_points", "points_diff",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_form_pts", "away_form_pts", "form_diff",
    "home_win_streak", "away_win_streak", "streak_diff",
    "home_unbeaten_streak", "away_unbeaten_streak",
    "home_gs_ewm", "home_gc_ewm", "away_gs_ewm", "away_gc_ewm",
    "home_cs_rate", "away_cs_rate", "home_fts_rate", "away_fts_rate",
    "h2h_home_wins", "h2h_away_wins", "h2h_gd"
]


def _model_path(league_code: str) -> str:
    return os.path.join(MODELS_DIR, f"{league_code}_xgb.pkl")


def _train_and_save(league_code: str, matches: list):
    """Train XGBClassifiers for outcome and goals, and persist to disk."""
    df = build_training_dataframe(matches)

    if len(df) < MIN_MATCHES_REQUIRED:
        raise ValueError(
            f"Only {len(df)} usable training rows for {league_code}. "
            f"Need at least {MIN_MATCHES_REQUIRED}."
        )

    X = df[FEATURE_COLS]

    # ── 1. Outcome classifier (home win / draw / away win) ──────────────
    y_cls = df["result"]
    clf = XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    clf.fit(X, y_cls)

    # ── 2. Goals classifiers (0, 1, 2, 3+) ──────────────────────────────
    y_home = df["home_goals_cat"]
    y_away = df["away_goals_cat"]

    goal_clf_home = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    goal_clf_away = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    
    goal_clf_home.fit(X, y_home)
    goal_clf_away.fit(X, y_away)

    payload = {
        "clf": clf,
        "goal_clf_home": goal_clf_home,
        "goal_clf_away": goal_clf_away,
        "n_rows": len(df),
        "accuracy": 0.0, # Placeholder or can re-calculate if needed
    }
    with open(_model_path(league_code), "wb") as f:
        pickle.dump(payload, f)

    _model_cache[league_code] = payload
    return payload


def _load_model(league_code: str):
    if league_code in _model_cache:
        return _model_cache[league_code]
    path = _model_path(league_code)
    if os.path.exists(path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Backwards-compat: old pkls or without goal classifiers → return None to force retrain
        if "goal_clf_home" not in payload:
            return None
        _model_cache[league_code] = payload
        return payload
    return None


def _calculate_market_metrics(home_probs: np.ndarray, away_probs: np.ndarray) -> dict:
    """
    Calculate advanced markets from direct goal class probabilities [0, 1, 2, 3+].
    """
    # 0: 0 goals, 1: 1 goal, 2: 2 goals, 3: 3+ goals
    p_h0, p_h1, p_h2, p_h3 = home_probs
    p_a0, p_a1, p_a2, p_a3 = away_probs

    # 1. BTTS (Both Teams To Score) = (1 - P(H0)) * (1 - P(A0))
    # Note: This assumes independence, which is standard for these calculations.
    prob_btts = (1.0 - p_h0) * (1.0 - p_a0)

    # 2. Over 2.5 goals
    # Possible combinations for Under 2.5: (0-0, 0-1, 0-2, 1-0, 1-1, 2-0)
    prob_under_25 = (
        (p_h0 * p_a0) + (p_h0 * p_a1) + (p_h0 * p_a2) +
        (p_h1 * p_a0) + (p_h1 * p_a1) +
        (p_h2 * p_a0)
    )
    prob_over_25 = 1.0 - prob_under_25

    # 3. Clean Sheets
    prob_cs_home = float(p_a0)
    prob_cs_away = float(p_h0)

    return {
        "btts": round(prob_btts * 100, 1),
        "over_2_5": round(prob_over_25 * 100, 1),
        "under_2_5": round(prob_under_25 * 100, 1),
        "clean_sheet_local": round(prob_cs_home * 100, 1),
        "clean_sheet_visitante": round(prob_cs_away * 100, 1)
    }


def _predict_top_scorelines(home_probs: np.ndarray, away_probs: np.ndarray, top_n: int = 5) -> list[dict]:
    """Calculate most probable scorelines from multiclass probabilities."""
    results = []
    # Use labels 0, 1, 2, 3 for 0, 1, 2, 3+
    labels = [0, 1, 2, 3]
    for i, h_label in enumerate(labels):
        for j, a_label in enumerate(labels):
            p = float(home_probs[i] * away_probs[j])
            h_display = str(h_label) if h_label < 3 else "3+"
            a_display = str(a_label) if a_label < 3 else "3+"
            results.append({"local": h_display, "visitante": a_display, "probabilidad": round(p * 100, 1)})
    
    results.sort(key=lambda x: x["probabilidad"], reverse=True)
    return results[:top_n]


def predict_match(league_code: str, home_team_id: int, away_team_id: int) -> dict:
    """
    Predict match outcome, expected goals, goal distributions and top scorelines.
    """
    matches = get_historical_matches(league_code)

    payload = _load_model(league_code)
    if payload is None:
        if len(matches) < 10:
            raise ValueError(
                f"No hay suficientes partidos finalizados para {league_code}. "
                "Intenta más tarde en la temporada."
            )
        logger.info(f"[{league_code}] No model found. Training now...")
        payload = _train_and_save(league_code, matches)

    clf: XGBClassifier = payload["clf"]
    g_clf_h: XGBClassifier = payload["goal_clf_home"]
    g_clf_a: XGBClassifier = payload["goal_clf_away"]
    n_rows: int = payload["n_rows"]

    feat_df = build_prediction_features(home_team_id, away_team_id, matches)
    X_pred  = feat_df[FEATURE_COLS]

    # ── Outcome probabilities ────────────────────────────────────────────
    proba    = clf.predict_proba(X_pred)[0]   # [away, draw, home]
    prob_away = float(proba[0]) * 100
    prob_draw = float(proba[1]) * 100
    prob_home = float(proba[2]) * 100

    # ── Goal class probabilities [0, 1, 2, 3+] ───────────────────────────
    # We ensure we have 4 classes. If league has few data, it might have fewer classes.
    # We'll pad with zeros if necessary (though rare in full seasons)
    def _get_full_probs(model, X):
        p = model.predict_proba(X)[0]
        full_p = np.zeros(4)
        for idx, class_val in enumerate(model.classes_):
            if int(class_val) < 4:
                full_p[int(class_val)] = p[idx]
        return full_p

    probs_h = _get_full_probs(g_clf_h, X_pred)
    probs_a = _get_full_probs(g_clf_a, X_pred)

    # ── Market Metrics (BTTS, Over/Under, etc.) ─────────────────────────
    markets = _calculate_market_metrics(probs_h, probs_a)

    # ── Top scorelines ───────────────────────────────────────────────────
    scorelines = _predict_top_scorelines(probs_h, probs_a)

    # ── Goal distribution list for UI ────────────────────────────────────
    def _fmt_dist(p_array):
        return [{"goles": str(i) if i < 3 else "3+", "probabilidad": round(p_array[i]*100, 1)} for i in range(4)]

    # ── Confidence ──────────────────────────────────────────────────────
    confidence = "Alta" if n_rows >= 100 else ("Media" if n_rows >= 50 else "Baja")

    return {
        "probabilidades": {
            "local":     round(prob_home, 2),
            "empate":    round(prob_draw, 2),
            "visitante": round(prob_away, 2),
        },
        "metricas_mercado": markets,
        "distribucion_goles": {
            "local":     _fmt_dist(probs_h),
            "visitante": _fmt_dist(probs_a),
        },
        "marcadores_probables": scorelines,
        "modelo_info": {
            "partidos_entrenados":  n_rows,
            "confianza":            confidence,
            "tipo":                 "XGBoost Multiclase (Directo)",
        },
    }
    
# Trigger reload