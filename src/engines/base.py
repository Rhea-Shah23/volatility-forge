# base class for option pricing engines

#imports
from abc import ABC, abstractmethod 
from dataclasses import dataclass 
from typing import Dict, Optional 
import numpy as np 

@dataclass 
class OptionContract: 
    # represents an options contract
    spot: float # current underlying price 
    strike: float # strike price 
    time_to_maturity: float # time to expriration (years)
    risk_free_rate: float #risk-free interest rate 
    volatility: float
    option_type: str 
    dividend_yield: float = 0.0 

    def __post_init__(self):
        if self.option_type.lower() not in ["call", "put"]: 
            raise ValueError("option_type must be either 'call' or 'put'")
        self.option_type = self.option_type.lower() 

@dataclass 
class PricingResult:
    # contains pricing results and greeks 
    price: float 
    delta: Optional[float] = None 
    gamma: Optional[float] = None 
    vega: Optional[float] = None 
    theta: Optional[float] = None 
    rho: Optional[float] = None 

    def to_dict(self) -> Dict[str, float]:
        #convert to dict 
        return {
            "price": self.price, 
            "delta": self.delta,
            "gamma": self.gamma, 
            "vega": self.vega, 
            "theta": self.theta, 
            "rho": self.rho
        }
    
class PricingEngine(ABC):
    # abstract base class for all pricing engines 
    @abstractmethod 
    def price(self, option: OptionContract) -> PricingResult:
        pass

    @property 
    @abstractmethod
    def name(self) -> str: 
        pass 