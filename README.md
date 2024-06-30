# Options Pricing Program

This Python program provides functionality to price European call and put options using two different methods: the Black-Scholes model and Monte Carlo simulations.

## Black-Scholes Model

The Black-Scholes model is a widely used method for pricing European options under certain assumptions:

### Parameters:
- **Spot Price (spot)**: Current price of the underlying asset.
- **Strike Price (strike)**: Price at which the option can be exercised.
- **Time to Maturity (maturity)**: Time until the option expires, in years.
- **Risk-free Rate (risk_free_rate)**: Annual risk-free interest rate, continuously compounded.
- **Volatility (volatility)**: Standard deviation of the asset's returns over the given time period.
- **Dividend Yield (dividend)**: Continuous dividend yield of the asset.

### Functions:
- **Call Value (`call_value()`)**: Calculates the price of a European call option.
- **Put Value (`put_value()`)**: Calculates the price of a European put option.
- **Greeks**: Delta (`delta_call()` and `delta_put()`), Gamma (`gamma()`), Vega (`vega()`), Rho (`rho_call()` and `rho_put()`), and Theta (`theta_call()`).

## Monte Carlo Simulation

Monte Carlo simulations provide a probabilistic approach to estimate option prices, especially useful for complex derivatives where analytical solutions are not feasible:

### Parameters:
- **Number of Time Steps (N)**: Number of time intervals for simulation.
- **Number of Simulations (M)**: Number of paths simulated to estimate option value.
- **Market Value (market_value)**: Initial market price of the option.
- **Time to Maturity (T)**: Time until the option expires, in years.

### Functions:
- **Call Value (`call_value()`)**: Uses Monte Carlo simulation to estimate the price of a European call option.
- **Put Value (`put_value()`)**: Uses Monte Carlo simulation to estimate the price of a European put option.

### Additional Information:
- **Accuracy**: Increasing the number of simulations (`M`) improves the accuracy of Monte Carlo estimates.
- **Risk-Neutral Pricing**: Monte Carlo simulations use risk-neutral pricing to estimate the expected payoff of options.

## Dependencies:
- **pandas_datareader**: Used to fetch historical stock prices for volatility estimation.
- **matplotlib**: Used for plotting results.
- **scipy.stats**: Provides statistical functions for probability distributions.

## Example Usage:
```python
# Example of Monte Carlo simulation for call option
Monte_Carlo_test = MonteCarlo()
Monte_Carlo_test.call_value()

# Example of Black-Scholes model for call option
Black_Scholes_test = BlackScholes()
Black_Scholes_test.call_value()
```

### Note:
- Ensure proper initialization of parameters (`spot`, `strike`, `maturity`, `risk_free_rate`, `volatility`, etc.) according to the specific option being priced.

