from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime 
from pandas_datareader import data as pdr
import scipy.stats as stats
from scipy.stats import norm

#option pricing model - BlackScholes Method
@dataclass
class BlackScholes:
    spot: float = 5481
    strike: float= 4400
    maturity = ((datetime.date(2024,11,21)-datetime.date.today()).days+1)/365 
    risk_free_rate: float= 0.044
    volatility: float= 0.3675
    dividend: float= 0
    
    def d1(self) -> float:
        return (np.log(self.spot / self.strike) + (self.risk_free_rate - self.dividend + self.volatility**2 / 2)
                * self.maturity) / (self.volatility * np.sqrt(self.maturity))
    
    def d2(self) -> float:
        return self.d1() - self.volatility * np.sqrt(self.maturity)
    
    def call_value(self) -> float:
        call_payoff = self.spot * np.exp(-self.dividend * self.maturity) * norm.cdf(self.d1(), 0, 1) - \
            self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(self.d2(), 0, 1)
        print(f'Call Value by Black-Scholes-Merton method is ${call_payoff}')
            
    def put_value(self) -> float:
        put_payoff = self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-self.d2(), 0, 1) - \
            self.spot * np.exp(-self.dividend * self.maturity) * norm.cdf(-self.d1(), 0, 1)
        print(f'Put Value by Black-Scholes-Merton method is ${put_payoff}')

