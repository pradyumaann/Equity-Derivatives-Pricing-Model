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

# Calculation of greeks    
    def delta_call(self) -> float:
        return norm.cdf(self.d1(), 0, 1)
    
    def delta_put(self) -> float:
        return -norm.cdf(-self.d1())
    
    def gamma(self) -> float:
        return norm.pdf(self.d1()) / (self.spot * self.volatility * np.sqrt(self.maturity))
    
    def vega(self) -> float:
        return self.spot * np.sqrt(self.maturity) * norm.pdf(self.d1()) * 0.01
    
    def rho_call(self) -> float:
        return  - (self.spot * norm.pdf(self.d1()) * self.volatility / (2 * np.sqrt(self.maturity)) - self.risk_free_rate * self.strike * np.exp(-self.risk * self.maturity) * norm.cdf(self.d2())) / 100
    
    def rho_put(self) -> float:
        return  - (self.spot * norm.pdf(self.d1()) * self.volatility / (2 * np.sqrt(self.maturity)) + self.risk_free_rate * self.strike * np.exp(-self.risk * self.maturity) * norm.cdf(self.d2())) / 100
    
    def theta_call(self) -> float:
        return - (self.spot * norm.pdf(self.d1()) * self.volatility / (2* np.sqrt(self.maturity)) - self.risk_free_rate * self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(self.d2())) / 365
    
# Valuation of Financial Derivatives through Monte Carlo Simulations is only possible by using Risk-Neutral Pricing and simulating risk-neutral asset paths
# In case of financial derivatives Monte Carlo is a powerful tool to price complex derivatives for which an analytical formula is not possible.    
# Monte Carlo simulation provides an easy way to deal with multiple random factors and the incorporation of more realistic asset price processes such as jumps in asset prices

# Option pricing model - Monte Carlo method
class MonteCarlo:
    
    S: float = 5481                                                            #stock price
    K: float = 4400                                                            #strike price
    vol: float = 0.3675                                                        #volatility (%)
    r: float = 0.044                                                           #risk-free rate (%)
    N: float = 10                                                              #number of time steps
    M: float = 1000000                                                         #number of simulations
    market_value: float = 1181                                                 #market price of option
    T = ((datetime.date(2024,11,29)-datetime.date.today()).days+1)/365         #time in years
    # To increase the accuracy of the model we can simply increase the number of Simulations
    # Risk Neutral pricing metodology tells us that: value of an option = risk-neutral expectation of its dicounted payoff,
    # we can estimate this expectation by computing the average of a large number of discounted payoff for a particular simulation
        
    def call_value(self) -> float:   
        # Precompute the constants
        dt = self.T/self.N
        nudt = (self.r-0.5*self.vol**2) * dt
        volsdt = self.vol * np.sqrt(dt)
        lnS = np.log(self.S)
    
        # Standar Error placeholders
        sum_CT = 0
        sum_CT2 = 0
    
        for i in range(self.M):
            lnSt = lnS
            for j in range(self.N):
                lnSt = lnSt + nudt + volsdt*np.random.normal()
            
            ST = np.exp(lnSt)
            CT = max(0, ST - self.K)
            sum_CT = sum_CT + CT
            sum_CT2 = sum_CT2 + CT*CT
            
        C0 = np.exp(-self.r*self.T)*sum_CT/self.M
        sigma = np.sqrt((sum_CT2 - sum_CT*sum_CT/self.M)*np.exp(-2*self.r*self.T) / (self.M - 1))
        SE = sigma/np.sqrt(self.M)
        print(f'Call Value using Monte-Carlo method is ${C0} with SE +/-{SE}')
        
        
