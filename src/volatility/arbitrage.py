# arbitrage detection; constraint enforcement for volatility surfaces 

import numpy as np 
from typing import List, Tuple 

# detects arbitrate opportunities in volaility surfaces
class ArbitrageDetector:
    @staticmethod 
    def check_calendar_arbitrage(maturities: np.ndarray, total_variances: np.ndarray, tolerance: float = 1e-6) -> bool:
        # check if total variance in non decreasing with maturity 
        for i in range(len(maturities) - 1):
            if total_variances[i+1] < total_variances[i] - tolerance:
                return False
        return True
    
    @staticmethod 
    def check_butterfly_arbitrage(strikes: np.ndarray, call_prices: np.ndarray, tolerance: float = 1e-6) -> bool:
        # check butterfly spread in arbitrage conditions 
        if len(strikes) < 3:
            return True 
        
        for i in range(1, len(strikes) - 1):
            K1, K2, K3 = strikes[i-1], strikes[i], strikes[i+1]
            C1, C2, C3 = call_prices[i-1], call_prices[i], call_prices[i+1]

            w1 = (K3 - K2) / (K3 - K1)
            w2 = -1.0 
            w3 = (K2 - K1) / (K3 - K1)

            butterfly_value = w1 * C1 + w2 * C2 + w3 * C3

            if butterfly_value < -tolerance:
                return False 

        return True 
    
    @staticmethod 
    def check_vertical_spread_arbitrage(strikes: np.ndarray, call_prices: np.ndarray, tolerance: float = 1e-6) -> bool:
        # check that call prices area decreasing with strike 
        for i in range(len(strikes) - 1):
            if call_prices[i+1] > call_prices[i] + tolerance:
                return False 
        return True 
    
    @staticmethod 
    def check_density_positivity(log_moneyness: np.ndarray, implied_vols: np.ndarray, maturity: float, num_points: int = 100) -> bool:
        # check implied probability density is positive 
        w = implied_vols**2 * maturity 

        if len(log_moneyness) < 3:
            return True 
        
        from scipy.interplolate import UnivariateSpline 
        try: 
            spline = UnivariateSpline(log_moneyness, w, k = 3, s = 0)
            k_dense = np.linspace(log_moneyness.min(), log_moneyness.max(), num_points)
            second_deriv = spline.derivative(n = 2)(k_dense)

            return np.all(second_deriv > -1e-6)
        
        except:
            return True 
        
    def check_surface(self, strikes: np.ndarray, maturities: np.ndarray, call_prices: np.ndarray) -> dict:
        results = {
            "abitrage_free": True,
            "violations": []
        }

        for i, T in enumerate(maturities):
            K = strikes[i] 
            C = call_prices[i] 

            if not self.check_butterfly_arbitrage(K, C):
                    results['arbitrage_free'] = False
                    results['violations'].append(f"Butterfly arbitrage at T={T}")
                
            if not self.check_vertical_spread_arbitrage(K, C):
                results['arbitrage_free'] = False
                results['violations'].append(f"Vertical spread arbitrage at T={T}")
            
        return results

            
