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
    

# complete volatility surface across strikes and maturity 
class VolatilitySurface: 
    def __init__(self):
        self.maturities = []
        self.svi_models = {} 

    # calibrate surface to market quotes
    def calibrate(self, market_quotes: List[MarketQuote]):
        # group by maturity 
        maturity_groups = {}
        for quote in market_quotes: 
            T = quote.maturity 
            if T not in maturity_groups:
                maturity_groups[T] = [] 
            maturity_groups[T].append(quote)

        # calibrate svi for each maturity 
        from ..engines.black_scholes import BlackScholesEngine
        bs_engine = BlackScholesEngine() 

        for T, quotes in maturity_groups.items():
            strikes = np.array([q.strike for q in quotes]) 
            spot = quotes[0].spot 
            rate = quotes[0].rate
            forward = spot * np.exp(rate * T)

            implied_vols = [] 
            for q in quotes:
                try:
                    from ..engines.base import OptionContract
                    option = OptionContract(
                        spot=q.spot,
                        strike=q.strike,
                        time_to_maturity=q.maturity,
                        risk_free_rate=q.rate,
                        volatility=0.2,
                        option_type=q.option_type
                    )
                    iv = bs_engine.implied_volatility(option, q.price)
                    implied_vols.append(iv)
                except:
                    implied_vols.append(0.2)

            implied_vols = np.array(implied_vols) 

            svi = SVI()
            try:
                svi.calibrate(strikes, np.full(len(strikes), T), spot, forward, implied_vols) 
                self.svi_models
                self.maturities.append
            except Exception as e:
                print(f"warning: svi calibration failed for T = {T}: {e}")

        self.maturities = sorted(self.maturities) 

    def get_volatility(self, strike: float, maturity: float, spot: float, rate: float) -> float:
        if not self.maturities:
            raise ValueError("surface not calibrated")
        
        forward = spot * np.exp(rate * maturity) 

        if maturity in self.svi_models:
            return self.svi_models[maturity].get_volatility(strike, forward)
        
        maturities = np.array(self.maturities) 

        if maturity < maturities[0]:
            return self.svi_models[maturities[0]].get_volatility(strike, forward) 
        elif maturity > maturities[-1]:
            return self.svi_models[maturities[-1]].get_volatility(strike, forward)
        else:
            idx = np.searchsorted(maturities, maturity)
            T1, T2 = maturities[idx - 1], maturities[idx]
            vol1 = self.svi_models[T1].get_volatility(strike, forward)
            vol2 = self.svi_models[T2].get_volatility(strike, forward)

            var1, var2 = vol1**2 * T1, vol2**2 * T2 
            var_interp = var1 + (var2 - var1) * (maturity - T1) / (T2 - T1)

            return np.sqrt(var_interp / maturity)