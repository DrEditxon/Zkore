import logging
import concurrent.futures
from app.services.data_service import data_service
from app.core.pipeline import predict_match

logger = logging.getLogger(__name__)

class HistoryService:
    def get_league_history(self, league_code: str, limit: int = 10):
        """
        Retrieves the last finished matches, predicts them, and compares results.
        """
        # 1. Get historical matches (from football-data.org or similar)
        # For consistency with the existing data_service, we use get_historical_matches
        matches_raw = data_service.get_historical_matches(league_code)
        
        # Sort by date descending and take the limit
        matches_raw.sort(key=lambda x: x['utcDate'], reverse=True)
        recent_matches = matches_raw[:limit]
        
        history_results = []
        hits = 0
        misses = 0
        
        def process_match(m):
            try:
                # Predict as if it hadn't happened
                # Note: This might be slightly biased if the model was trained on these matches,
                # but it serves the user's purpose of "visualizing accuracy".
                prediction = predict_match(
                    league_code,
                    m['homeTeam_id'],
                    m['awayTeam_id'],
                    m['homeTeam_name'],
                    m['awayTeam_name'],
                    match_id=m["match_id"],
                    utc_date=m["utcDate"]
                )
                
                # Actual Result
                actual_home = m['homeGoals']
                actual_away = m['awayGoals']
                
                if actual_home > actual_away:
                    actual_verdict = "L"
                elif actual_away > actual_home:
                    actual_verdict = "V"
                else:
                    actual_verdict = "E"
                    
                # Predicted Result (highest probability)
                probs = prediction['probabilidades']
                if probs['local'] > probs['empate'] and probs['local'] > probs['visitante']:
                    pred_verdict = "L"
                elif probs['visitante'] > probs['empate'] and probs['visitante'] > probs['local']:
                    pred_verdict = "V"
                else:
                    pred_verdict = "E"
                
                # Check Winner Hit
                winner_hit = (actual_verdict == pred_verdict)
                
                # Check Goals Hit (Total goals within +/- 1 of predicted xG total?)
                # Or compare with the most probable scoreline
                predicted_xg_total = prediction['expected_goals']['local'] + prediction['expected_goals']['visitante']
                actual_goals_total = actual_home + actual_away
                goals_hit = abs(predicted_xg_total - actual_goals_total) <= 1.0
                
                # Cards and Shots (Mocked for now as we don't have historical per-match stats easily)
                # In a real scenario, we'd fetch these from RapidAPI /fixtures/statistics
                # For this demo, we'll mark them based on proximity if we had them.
                # Since we don't have them in 'm', we'll just focus on winner for the 'Hit/Miss' count
                # but we'll include them in the details if we can fetch them.
                
                return {
                    "match": f"{m['homeTeam_name']} vs {m['awayTeam_name']}",
                    "date": m['utcDate'],
                    "actual_score": f"{actual_home}-{actual_away}",
                    "prediction": {
                        "winner": pred_verdict,
                        "probs": probs,
                        "xg": prediction['expected_goals']
                    },
                    "is_hit": winner_hit,
                    "details": {
                        "winner_hit": winner_hit,
                        "goals_hit": goals_hit,
                    }
                }
            except Exception as e:
                logger.error(f"Error processing history for match: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_match, recent_matches))
            
        history_results = [r for r in results if r is not None]
        hits = sum(1 for r in history_results if r['is_hit'])
        misses = len(history_results) - hits
        
        return {
            "summary": {
                "hits": hits,
                "misses": misses,
                "total": len(history_results),
                "accuracy": round((hits / len(history_results)) * 100, 1) if history_results else 0
            },
            "history": history_results
        }

history_service = HistoryService()
