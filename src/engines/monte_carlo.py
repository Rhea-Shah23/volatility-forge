# monte carlo simulation engine 
# variance reduction techniques 

import numpy as np 
from .base import PricingEngine, OptionContract, PricingResult 

# monte carlo simulation for option pricing 
# features: antithetic variates, quasi-random numbers, pathwise derivatives 
class MonteCarloEngine(PricingEngine):
    def __init__(self, n_simulations: int = 10000, antithetic: bool = True, quasi_random: bool = False, seed: int = 22):
        self.n_simulations = n_simulations 
        self.antithetic = antithetic 
        self.quasi_random = quasi_random 
        self.seed = seed 
        np.random.seed(seed) 

    @property 
    def name(self) -> str: 
        variance_reduction = [] 
        if self.antithetic:
            variance_reduction.append("antithetic")
        if self.quasi_random:
            variance_reduction.append("sobol") 

        vr_str = "-".join(variance_reduction) if variance_reduction else "standard" 
        return f"Monte Carlo - {vr_str}-{self.n_simulations}" 
    
    def price(self, option: OptionContract) -> PricingResult: 
        S = option.spot 
        K = option.strike 
        T = option.time_to_maturity
        r = option.risk_free_rate 
        sigma = option.volatility 
        q = option.dividend_yield 

        if T <= 0: 
            if option.option_type == "call":
                return PricingResult(price = max(S - K, 0)) 
            else: 
                return PricingResult(price = max(K - S, 0)) 
            
        # generate random numbers
        if self.quasi_random: 
            # sobol sequence 
            Z = np.random.randn(self.n_simulations) 
        else: 
            Z = np.random.randn(self.n_simulations)

        # simulate terminal stock prices
        drift = (r - q - 0.5 * sigma**2) * T 
        diffusion = sigma * np.sqrt(T) 
        ST = S * np.exp(drift + diffusion * Z) 

        # calc payoffs 
        if option.option_type == "call":
            payoffs = np.maximum(ST - K, 0) 
        else: 
            payoffs = np.maximum(K - ST, 0) 

        # antithetic variates 
        if self.antithetic:
            ST_anti = S * np.exp(drift - diffusion * Z) 
            if option.option_type == "call":
                payoffs_anti = np.maximum(ST_anti - K, 0) 
            else: 
                payoffs_anti = np.maximum(K - ST_anti, 0)

            payoffs = (payoffs + payoffs_anti) / 2 

        price = np.exp(-r * T) * np.mean(payoffs) 

        # calc delta; pathwise method 
        if option.option_type == "call":
            indicators = (ST > K).astype(float) 
            delta = np.exp(-q * T) * np.mean(indicators) 
        else:
            indicators = (ST < K).astype(float) 
            delta = -np.exp(-q * T) * np.mean(indicators) 

        std_error = np.std(payoffs) / np.sqrt(self.n_simulations) 

        return PricingResult(price = price, delta = delta) 