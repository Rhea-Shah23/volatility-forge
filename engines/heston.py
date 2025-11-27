# heston stochastic volatility model 

import numpy as np 
from scipy.integrate import quad 
from .base import PricingEngine, OptionContract, PricingResult 

# heston stochastic volatility model; fourier inversion 
class HestonEngine(PricingEngine):
    def __init__(self, kappa: float = 2.0, theta: float = 0.04, sigma_v: float = 0.3, rho: float = -0.7, v0: float = 0.04):
        self.kappa = kappa 
        self.theta = theta 
        self.sigma_v = sigma_v 
        self.rho = rho 
        self.v0 = v0

        # feller condition check 
        if 2 * kappa * theta < sigma_v**2:
            raise ValueError("Feller condition not satisfied; variance can reach zero")
        
    @property 
    def name(self) -> str:
        return "Heston-StochasticVol" 
    
    def price(self, option: OptionContract) -> PricingResult: 
        S = option.spot
        K = option.strike
        T = option.time_to_maturity
        r = option.risk_free_rate
        q = option.dividend_yield
        
        if T <= 0:
            if option.option_type == 'call':
                return PricingResult(price=max(S - K, 0))
            else:
                return PricingResult(price=max(K - S, 0))
        
        # characteristic function approach
        def characteristic_function(phi, j):
            if j == 1:
                u, b = 0.5, self.kappa - self.rho * self.sigma_v
            else:
                u, b = -0.5, self.kappa
            
            a = self.kappa * self.theta
            x = np.log(S)
            
            d = np.sqrt((self.rho * self.sigma_v * phi * 1j - b)**2 
                       - self.sigma_v**2 * (2 * u * phi * 1j - phi**2))
            g = (b - self.rho * self.sigma_v * phi * 1j + d) / \
                (b - self.rho * self.sigma_v * phi * 1j - d)
            
            C = (r - q) * phi * 1j * T + (a / self.sigma_v**2) * \
                ((b - self.rho * self.sigma_v * phi * 1j + d) * T - 
                 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
            
            D = ((b - self.rho * self.sigma_v * phi * 1j + d) / self.sigma_v**2) * \
                ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
            
            return np.exp(C + D * self.v0 + 1j * phi * x)
        
        # probability function
        def P(j):
            def integrand(phi):
                cf = characteristic_function(phi, j)
                return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))
            
            integral, _ = quad(integrand, 1e-10, 100, limit=100)
            return 0.5 + (1 / np.pi) * integral
        
        # calc probabilities 
        P1 = P(1)
        P2 = P(2)
        
        # option price
        if option.option_type == 'call':
            price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
        else:  # put
            price = K * np.exp(-r * T) * (1 - P2) - S * np.exp(-q * T) * (1 - P1)
        
        # approx delta
        if option.option_type == 'call':
            delta = np.exp(-q * T) * P1
        else:
            delta = -np.exp(-q * T) * (1 - P1)
        
        return PricingResult(price=price, delta=delta)