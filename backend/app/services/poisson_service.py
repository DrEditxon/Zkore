import numpy as np
from scipy.stats import poisson

class PoissonService:
    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals

    def calculate_probability_matrix(self, lambda_home: float, lambda_away: float, rho: float = 0.0) -> np.ndarray:
        """
        Generates a 2D matrix of shape (max_goals+1, max_goals+1) where
        M[i][j] is the probability of Home scoring i goals and Away scoring j.
        Includes Dixon-Coles adjustment parameter (rho).
        """
        lambda_h = max(0.01, float(lambda_home))
        lambda_a = max(0.01, float(lambda_away))
        
        prob_h = poisson.pmf(np.arange(self.max_goals + 1), lambda_h)
        prob_a = poisson.pmf(np.arange(self.max_goals + 1), lambda_a)
        
        matrix = np.outer(prob_h, prob_a)
        
        # Apply Dixon-Coles adjustment
        if rho != 0.0:
            tau = np.ones_like(matrix)
            tau[0, 0] = max(0, 1 - lambda_h * lambda_a * rho)
            tau[0, 1] = max(0, 1 + lambda_h * rho)
            tau[1, 0] = max(0, 1 + lambda_a * rho)
            tau[1, 1] = max(0, 1 - rho)
            matrix *= tau

            # Normalize after adjustment
            matrix = matrix / np.sum(matrix)
            
        return matrix

    def extract_metrics(self, prob_matrix: np.ndarray) -> dict:
        """
        Extracts complex metrics from the probability matrix.
        """
        prob_home_win = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away_win = np.sum(np.triu(prob_matrix, 1))

        # BTTS (Both Teams To Score) - exclude 0 goals row and column
        prob_btts = np.sum(prob_matrix[1:, 1:])

        # Over / Under 2.5
        prob_under_2_5 = 0.0
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                if i + j <= 2:
                    prob_under_2_5 += prob_matrix[i, j]
        prob_over_2_5 = 1.0 - prob_under_2_5

        # Clean Sheets
        prob_cs_home = np.sum(prob_matrix[:, 0])
        prob_cs_away = np.sum(prob_matrix[0, :])

        return {
            "prob_home_win": round(float(prob_home_win) * 100, 2),
            "prob_draw": round(float(prob_draw) * 100, 2),
            "prob_away_win": round(float(prob_away_win) * 100, 2),
            "btts": round(float(prob_btts) * 100, 1),
            "over_2_5": round(float(prob_over_2_5) * 100, 1),
            "under_2_5": round(float(prob_under_2_5) * 100, 1),
            "clean_sheet_local": round(float(prob_cs_home) * 100, 1),
            "clean_sheet_visitante": round(float(prob_cs_away) * 100, 1)
        }

    def get_top_scorelines(self, prob_matrix: np.ndarray, top_n: int = 5) -> list[dict]:
        """
        Returns the top N most probable scorelines.
        """
        results = []
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                p = prob_matrix[i, j]
                h_display = str(i) if i < self.max_goals else f"{self.max_goals}+"
                a_display = str(j) if j < self.max_goals else f"{self.max_goals}+"
                results.append({"local": h_display, "visitante": a_display, "probabilidad": round(float(p) * 100, 1)})
        
        results.sort(key=lambda x: x["probabilidad"], reverse=True)
        return results[:top_n]

    def format_goal_distributions(self, lambda_home: float, lambda_away: float) -> dict:
        """
        Formats the marginal distributions for the UI (0 up to 3+).
        """
        lambda_h = max(0.01, float(lambda_home))
        lambda_a = max(0.01, float(lambda_away))
        
        prob_h = poisson.pmf(np.arange(4), lambda_h)
        prob_h[3] += 1.0 - np.sum(prob_h) # Aggregate 3 and above into the 3 index
        
        prob_a = poisson.pmf(np.arange(4), lambda_a)
        prob_a[3] += 1.0 - np.sum(prob_a)
        
        def _fmt_dist(p_array):
            return [{"goles": str(i) if i < 3 else "3+", "probabilidad": round(float(p_array[i])*100, 1)} for i in range(4)]

        return {
            "local": _fmt_dist(prob_h),
            "visitante": _fmt_dist(prob_a)
        }

poisson_service = PoissonService()
