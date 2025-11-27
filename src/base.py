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

