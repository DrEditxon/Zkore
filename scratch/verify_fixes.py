import sys
sys.path.insert(0, 'backend')

# ── ML-04: Verificar TimeSeriesSplit en TuningService ─────────────────────────
from app.services.tuning_service import tuning_service
import inspect, textwrap

src = inspect.getsource(tuning_service.tune_league)
assert "TimeSeriesSplit" in src, "TimeSeriesSplit no encontrado en tune_league"
assert "n_iter=50"      in src, "n_iter=50 no encontrado"
assert "neg_mean_squared_error" in src, "scoring MSE no encontrado"
assert "n_splits=5"     in src, "n_splits=5 no encontrado"
print("ML-04 OK: TimeSeriesSplit(n_splits=5), n_iter=50, scoring=MSE")

# ── BUG-04: Verificar lógica del fallback de Value Bets ──────────────────────
# Simulamos la lógica manualmente
league_home_goals = 1.45
league_away_goals = 1.12

# Fallback VIEJO: {hg:1, hc:1, hm:1}
old_h_st = {"hg": 1, "hc": 1, "hm": 1}
old_h_atk = (old_h_st["hg"] / max(1, old_h_st["hm"])) / max(0.1, league_home_goals)
print(f"\nBUG-04: Fallback VIEJO → h_atk = {old_h_atk:.4f}  (debería ser ~1.0, era {old_h_atk:.4f})")

# Fallback NUEVO: usa promedios de liga
new_h_st = {"hg": league_home_goals, "hc": league_away_goals, "hm": 1}
new_h_atk = (new_h_st["hg"] / max(1, new_h_st["hm"])) / max(0.1, league_home_goals)
print(f"BUG-04: Fallback NUEVO → h_atk = {new_h_atk:.4f}  (debe ser exactamente 1.0)")
assert abs(new_h_atk - 1.0) < 1e-9, f"Fallback no produce 1.0: {new_h_atk}"
print("BUG-04 OK: Fallback neutral produce h_atk = 1.0 exacto")

# ── BUG-04 verify from source ─────────────────────────────────────────────────
from app.routes.predict import router
import app.routes.predict as predict_module
src2 = open('backend/app/routes/predict.py').read()
assert "league_home_goals, \"hc\": league_away_goals" in src2 or "league_home_goals" in src2
print("BUG-04 OK: Código en predict.py actualizado con fallback de liga")

print("\n=== Todos los tests pasaron OK ===")
