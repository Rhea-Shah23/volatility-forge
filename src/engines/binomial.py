# binomial tree ricing engine
# american option support 

# imports
import numpy as np 
from .base import PricingEngine, OptionContract, PricingResult 

class BinomialTreeEngine(PricingEngine):
    # cox-ross-rubinstein (CRR) binomial tree model 
    def __init__(self, steps: int = 100, american: bool = False): 
        # initialize binomial tree engine 
        self.steps = steps 
        self.american = american 

    @property 
    def name(self) -> str: 
        exercise = "American" if self.american else "European" 
        return f"Binomial Tree -{exercise} - {self.steps}" 
    
    def price(self, option: OptionContract) -> PricingResult: 
        # price using binomial tree 
        S = option.spot 
        K = option.strike 
        T = option.time_to_maturity
        r = option.risk_free_rate
        sigma = option.volatility
        q = option.dividend_yield
        N = self.steps 

        if T <= 0:
            if option.option_type == "call": 
                return PricingResult(price = max(S-K, 0)) 
            else: 
                return PricingResult(price = max(K-S, 0)) 
            
        # time step 
        dt = T / N 

        # crr parameters 
        u = np.exp(sigma * np.sqrt(df)) 
        d = 1 / u 
        p = (np.exp((r - q) * dt) - d) / (u - d) 
        discount = np.exp(-r * dt) 

        # initialize asset prices @ maturity 
        ST = np.zeros(N + 1)
        for i in range(N + 1):
            ST[i] = S * (u ** (N - i)) * (d ** i) 

        # initialize option values @ maturity 
        if option.option_type == "call":
            option_values = np.maximum(ST - K, 0) 
        else:
            option_values = np.maximum(K - ST, 0) 

        # backward induction through tree 
        for step in range(N - 1, -1, -1): 
            for i in range(step + 1):
                # stock price @ node 
                S_node = S * (u ** (step - i)) * (d ** i) 

                # continuation value 
                continuation = discount * (p * option_values[i] + (1 - p) * option_values[i + 1]) 

                if self.american:
                    if option.option_type == "call":
                        exercise = max(S_node - K, 0) 
                    else: 
                        exercise = max(K - S_node, 0)
                    option_values[i] = max(continuation, exercise)
                else:
                    option_values[i] = continuation
        price = option_values[0]

        # calc delta using finite difference 
        if N > 1: 
            S_up = S * u 
            S_down = S * d 
            option_up = option_values[0]
            option_down = option_values[1] 

            delta = (option_up - option_down) / (S_up - S_down) 
        else:
            delta = None 

        return PricingResult(price = price, delta = delta) 