# Black-Scholes-Merton analytical pricing engine 

#import 
import numpy as np 
from scipy.stats import norm 
from .base import PricingEngine, OptionContract, PricingResult 

class BlackScholesEngine(PricingEngine):
    # closed-form solution 
    @property 
    def name(self) -> str:
        return "Black-Scholes" 
    
    def price (self, option: OptionContract) -> PricingResult:
        # price using Black-Scholes formula 
        S = option.spot 
        K = option.strike 
        T = option.time_to_maturity
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield

        # edge cases 
        if T <= 0: 
            if option.option_type == "call":
                return PricingResult(price = max(S - K, 0)) 
            else:
                return PricingResult(price = max(K - S, 0)) 
            
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        
        # calc d1 and d2
        d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # calc price 
        if option.option_type == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) 
            delta = np.exp(-q * T) * norm.cdf(d1)
        else: 
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            delta = -np.exp(-q * T) * norm.cdf(-d1)

        # calc greeks
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100 # per 1% vol 

        if option.option_type == 'call':
            theta = ((-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else: 
            theta = ((-S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return PricingResult(
            price=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho
        )