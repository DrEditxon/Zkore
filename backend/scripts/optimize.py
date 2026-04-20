import sys
import os
import json
import logging
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

# Añadir base dir a PATH para los imports relativos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.data_service import data_service
from app.services.feature_service import feature_service, FEATURE_COLS
from app.services.poisson_service import poisson_service

# Configuramos logging de optuna para no ensuciar mucho
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEAGUE_CODE = "PD"  # Liga base para entrenar los óptimos generales

def evaluate_predictions(y_true_h, y_true_a, y_pred_h, y_pred_a):
    mae_h = mean_absolute_error(y_true_h, y_pred_h)
    mae_a = mean_absolute_error(y_true_a, y_pred_a)
    return (mae_h + mae_a) / 2

def convert_to_wdl(h_goals, a_goals):
    res = []
    for h, a in zip(h_goals, a_goals):
        if h > a: res.append(2)  # Home
        elif h == a: res.append(1)  # Draw
        else: res.append(0)  # Away
    return np.array(res)

def get_poisson_probs(y_pred_h, y_pred_a):
    probs = []
    for h, a in zip(y_pred_h, y_pred_a):
        matrix = poisson_service.calculate_probability_matrix(h, a)
        metrics = poisson_service.extract_metrics(matrix)
        # Orden de clases: 0: Away, 1: Draw, 2: Home
        probs.append([metrics["prob_away_win"]/100, metrics["prob_draw"]/100, metrics["prob_home_win"]/100])
    return np.array(probs)

def main():
    print(f"--- Iniciando Optimización Avanzada sobre {LEAGUE_CODE} ---")
    matches = data_service.get_historical_matches(LEAGUE_CODE)
    if len(matches) < 30:
        print("No hay suficientes partidos históricos")
        return
        
    df = feature_service.build_training_dataframe(matches)
    X = df[FEATURE_COLS]
    y_h = df["target_home_goals"]
    y_a = df["target_away_goals"]
    
    # Apartamos los últimos 40 partidos para test independiente
    X_train, X_test = X.iloc[:-40], X.iloc[-40:]
    yh_train, yh_test = y_h.iloc[:-40], y_h.iloc[-40:]
    ya_train, ya_test = y_a.iloc[:-40], y_a.iloc[-40:]
    
    # ----------------------------------------------------
    # 1. BASELINE
    # ----------------------------------------------------
    model_params_base = {
        "n_estimators": 200, 
        "max_depth": 3, 
        "learning_rate": 0.05,
        "subsample": 0.8, 
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    baseline_h = XGBRegressor(**model_params_base)
    baseline_h.fit(X_train, yh_train)
    baseline_a = XGBRegressor(**model_params_base)
    baseline_a.fit(X_train, ya_train)
    
    pred_h_base = np.maximum(0.01, baseline_h.predict(X_test))
    pred_a_base = np.maximum(0.01, baseline_a.predict(X_test))
    
    base_mae = evaluate_predictions(yh_test, ya_test, pred_h_base, pred_a_base)
    
    wdl_test = convert_to_wdl(yh_test, ya_test)
    base_probs = get_poisson_probs(pred_h_base, pred_a_base)
    base_logloss = log_loss(wdl_test, base_probs, labels=[0, 1, 2])
    
    print(f"[ANTES] Baseline MAE:      {base_mae:.4f}")
    print(f"[ANTES] Baseline LOG LOSS: {base_logloss:.4f} (Probabilidades puras poisson)")

    # ----------------------------------------------------
    # 2. TUNING BAYESIANO CON OPTUNA
    # ----------------------------------------------------
    print(f"\nEjecutando Búsqueda Optuna (Bayesiana) - Espere...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "random_state": 42
        }
        
        kf = KFold(n_splits=3, shuffle=False)
        maes = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
            yh_tr, yh_va = y_h.iloc[tr_idx], y_h.iloc[val_idx]
            ya_tr, ya_va = y_a.iloc[tr_idx], y_a.iloc[val_idx]
            
            m_h = XGBRegressor(**params)
            m_h.fit(X_tr, yh_tr)
            m_a = XGBRegressor(**params)
            m_a.fit(X_tr, ya_tr)
            
            p_h = np.maximum(0.01, m_h.predict(X_va))
            p_a = np.maximum(0.01, m_a.predict(X_va))
            maes.append(evaluate_predictions(yh_va, ya_va, p_h, p_a))
            
        return np.mean(maes)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    best_params["random_state"] = 42
    print(f"-> Mejores Params: {best_params}")
    
    # ----------------------------------------------------
    # 3. FEATURE IMPORTANCE Y ELIMINACIÓN DE RUIDO
    # ----------------------------------------------------
    best_h = XGBRegressor(**best_params)
    best_h.fit(X_train, yh_train)
    best_a = XGBRegressor(**best_params)
    best_a.fit(X_train, ya_train)
    
    avg_imp = (best_h.feature_importances_ + best_a.feature_importances_) / 2
    feature_imp = pd.DataFrame({"feature": FEATURE_COLS, "importance": avg_imp}).sort_values("importance", ascending=False)
    
    # Desechar los que tengan < 0.01 o algo equivalente. Criterio de poda.
    # Dado que son pocos datos, ser muy agresivo puede romper el modelo base, usaremos < 0.005.
    zero_imp_features = feature_imp[feature_imp["importance"] <= 0.005]["feature"].tolist()
    pruned_features = [f for f in FEATURE_COLS if f not in zero_imp_features]
    
    print(f"-> Features Eliminados (Importancia <= 0.005): {zero_imp_features}")
    print(f"-> Features Retenidos: {len(pruned_features)}")
    
    # Entrenar modelo optimizado y podado final
    X_train_pruned = X_train[pruned_features]
    X_test_pruned = X_test[pruned_features]
    
    final_h = XGBRegressor(**best_params)
    final_h.fit(X_train_pruned, yh_train)
    final_a = XGBRegressor(**best_params)
    final_a.fit(X_train_pruned, ya_train)
    
    pred_h_final = np.maximum(0.01, final_h.predict(X_test_pruned))
    pred_a_final = np.maximum(0.01, final_a.predict(X_test_pruned))
    
    final_mae = evaluate_predictions(yh_test, ya_test, pred_h_final, pred_a_final)
    final_probs = get_poisson_probs(pred_h_final, pred_a_final)
    
    print(f"[DESPUÉS] MAE Optimizado & Podado: {final_mae:.4f}")
    
    # ----------------------------------------------------
    # 4. CALIBRACIÓN DE PROBABILIDADES
    # ----------------------------------------------------
    # Extraemos previas probabilísticas del set de entrenamiento usando out-of-fold o datos de train
    # Para simplicidad y robustez, usaremos entrenamiento (sobreajuste evitado x linear_model)
    X_calib = get_poisson_probs(np.maximum(0.01, final_h.predict(X_train_pruned)), 
                                np.maximum(0.01, final_a.predict(X_train_pruned)))
    y_calib = convert_to_wdl(yh_train, ya_train)
    
    # Regresión Logística Multi-Clase como calibrador Platt Scaling
    calibrator = LogisticRegression(max_iter=2000)
    calibrator.fit(X_calib, y_calib)
    
    # Predecimos el test post-calibración
    calib_test_probs = calibrator.predict_proba(final_probs)
    calib_logloss = log_loss(wdl_test, calib_test_probs, labels=[0, 1, 2])
    
    print(f"[DESPUÉS] Log-Loss Calibrado:    {calib_logloss:.4f}")
    
    # Guardamos los resultados
    report = {
        "best_params": best_params,
        "dropped_features": zero_imp_features,
        "pruned_features": pruned_features,
        "base_mae": base_mae,
        "optimized_mae": final_mae,
        "base_logloss": base_logloss,
        "calibrated_logloss": calib_logloss
    }
    with open(os.path.join(os.path.dirname(__file__), "optimization_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n--- Optimización Finalizada. Reporte guardado en scripts/optimization_report.json ---")

if __name__ == "__main__":
    main()
