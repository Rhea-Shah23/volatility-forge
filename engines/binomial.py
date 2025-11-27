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
    
    