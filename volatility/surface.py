#volatlity surface parameterization & construction 

import numpy as np 
from scipy.optimize import minimize, least_squares 
from typing import List, Tuple, Callable 
from dataclasses import dataclass 

@dataclass 
class MarketQuote:
    # market option quote for calibration 
    strike: float 
    maturity: float 
    option_type: str 
    price: float 
    spot: float
    rate: float 

class SVI:
    # stochastic volatility inspired parameterization 
    def __init__(self):
        self.params = None 

    def total_variance(self, log_moneyness: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray: 
        # calc total variance 
        k = log_moneyness
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2)) 
    
    def implied_volatility(self, log_moneyness: np.ndarray, maturity: float, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        w = self.total_variance(log_moneyness, a, b, rho, m ,sigma) 
        return np.sqrt(w / maturity) 
    
    def calibrate(self, strikes: np.ndarray, maturities: np.ndarray, spot: float, forward: float, implied_vols: np.ndarray) -> dict: 
        T = maturities[0] 
        log_moneyness = np.log(strikes / forward) 
        total_var_market = implied_vols**2 * T 

        # initial guess 
        a0 = 0.04 * T
        b0 = 0.1 
        rho0 = 0.0 
        m0 = 0.0 
        sigma0 = 0.1 

        x0 = np.array([a0, b0, rho0, m0, sigma0]) 

        def constraint_rho(x):
            return 1 - abs(x[2]) 
        
        def constraint_b(x):
            return x[1] 
        
        def constraint_sigma(x):
            return x[4] 
        
        def constraint_butterfly(x):
            a, b, rho, m, sigma = x 
            return 4 - b * (1 + abs(rho)) 
        
        constraints = [ 
            {'type': 'ineq', 'fun': constraint_rho},
            {'type': 'ineq', 'fun': constraint_b},
            {'type': 'ineq', 'fun': constraint_sigma},
            {'type': 'ineq', 'fun': constraint_butterfly}
        ]

        # objective function
        def objective(x):
            a, b, rho, m, sigma = x
            w_model = self.total_variance(log_moneyness, a, b, rho, m, sigma)
            return np.sum((w_model - total_var_market)**2)
        
        # optimize
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        if result.success:
            self.params = {
                'a': result.x[0],
                'b': result.x[1],
                'rho': result.x[2],
                'm': result.x[3],
                'sigma': result.x[4],
                'maturity': T
            }
            return self.params
        else:
            raise ValueError(f"SVI calibration failed: {result.message}")
        
    def get_volatility(self, strike: float, forward: float) -> float:
        if self.params is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        log_moneyness = np.log(strike / forward)
        T = self.params['maturity']
        
        vol = self.implied_volatility(
            np.array([log_moneyness]), T,
            self.params['a'], self.params['b'], self.params['rho'],
            self.params['m'], self.params['sigma']
        )
        
        return vol[0]