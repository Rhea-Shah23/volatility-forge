# merton jump diffusion model 

import numpy as np 
from scipy.stats import norm, poisson 
from .base import PricingEngine, OptionContract, PricingResult 

# merton jump diffusion model 
class JumpDiffusionEngine(PricingEngine):
    def __init__(self, lambda_jump: float = 0.1, mu_jump: float = -0.05, sigma_jump: float = 0.15, max_jumps: int = 50):
        self.lambda_jump = lambda_jump 
        self.mu_jump = mu_jump 
        self.sigma_jump = sigma_jump 
        self.max_jumps = max_jumps

    @property 
    def name(self) -> str:
        return "JumpDiffusion-Merton" 
    
    # price using merton's jump diffusion formula 
    def price(self, option: OptionContract) -> PricingResult:
        S = option.spot
        K = option.strike
        T = option.time_to_maturity
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield
        
        if T <= 0:
            if option.option_type == 'call':
                return PricingResult(price=max(S - K, 0))
            else:
                return PricingResult(price=max(K - S, 0))
        
        # jump-adjusted parameters
        lambda_T = self.lambda_jump * T
        k = np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1  # Expected jump size
        
        # sum over possible number of jumps
        price = 0.0
        for n in range(self.max_jumps):
            # probability of n jumps
            prob_n = poisson.pmf(n, lambda_T)
            
            if prob_n < 1e-10:
                break
            
            # adjusted parameters for n jumps
            sigma_n = np.sqrt(sigma**2 + n * self.sigma_jump**2 / T)
            r_n = r - self.lambda_jump * k + n * (self.mu_jump + 0.5 * self.sigma_jump**2) / T
            
            # black scholes price with adjusted parameters
            d1 = (np.log(S / K) + (r_n - q + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
            d2 = d1 - sigma_n * np.sqrt(T)
            
            if option.option_type == 'call':
                bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
            else:  # put
                bs_price = K * np.exp(-r_n * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
            price += prob_n * bs_price
        
        return PricingResult(price=price)