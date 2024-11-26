from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime 
import yfinance as yf
from scipy.stats import norm
import seaborn as sns

class OptionsAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        
        # Get current stock price more reliably
        try:
            # Try to get real-time price first
            self.spot_price = self.stock.fast_info['last_price']
        except:
            try:
                # Fallback to historical data
                self.spot_price = self.stock.history(period='1d')['Close'].iloc[-1]
            except:
                raise ValueError(f"Could not fetch price data for {ticker}")
                
        self.risk_free_rate = 0.044  
        print(f"Successfully initialized {ticker} with spot price: ${self.spot_price:.2f}")
        
    def calculate_greeks(self, row):
        """Calculate Greeks for a single option"""
        S = self.spot_price
        K = row['strike']
        T = row['tte']
        r = self.risk_free_rate
        sigma = row['impliedVolatility']
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            return pd.Series({
                'delta': 0,
                'gamma': 0,
                'theta': 0,
                'vega': 0,
                'rho': 0
            })
        
        # Calculate d1 and d2
        try:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
        except:
            return pd.Series({
                'delta': 0,
                'gamma': 0,
                'theta': 0,
                'vega': 0,
                'rho': 0
            })
        
        # Common terms
        nd1 = norm.pdf(d1)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        
        # Calculate Greeks
        if row['option_type'] == 'call':
            delta = Nd1
            theta = (-S*nd1*sigma/(2*np.sqrt(T)) - 
                    r*K*np.exp(-r*T)*Nd2)/365
            rho = K*T*np.exp(-r*T)*Nd2/100
        else:  # put
            delta = Nd1 - 1
            theta = (-S*nd1*sigma/(2*np.sqrt(T)) + 
                    r*K*np.exp(-r*T)*norm.cdf(-d2))/365
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
            
        gamma = nd1/(S*sigma*np.sqrt(T))
        vega = S*np.sqrt(T)*nd1/100  # Divided by 100 to get per 1% change
        
        return pd.Series({
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        })
    
    def get_options_chain(self):
        try:
            # Get all available expiration dates
            expirations = self.stock.options
            
            if not expirations:
                raise ValueError(f"No options data available for {self.ticker}")
            
            # Initialize lists to store data
            all_options = []
            
            for exp in expirations:
                try:
                    # Get option chain for this expiration
                    opt = self.stock.option_chain(exp)
                    
                    # Process calls
                    calls = opt.calls
                    calls['option_type'] = 'call'
                    
                    # Process puts
                    puts = opt.puts
                    puts['option_type'] = 'put'
                    
                    # Combine calls and puts
                    options = pd.concat([calls, puts])
                    
                    # Add expiration date
                    options['expiration'] = exp
                    
                    all_options.append(options)
                except Exception as e:
                    print(f"Warning: Could not process expiration {exp}: {str(e)}")
                    continue
            
            if not all_options:
                raise ValueError(f"Could not process any options data for {self.ticker}")
                
            # Combine all options data
            options_df = pd.concat(all_options, ignore_index=True)
            
            # Calculate time to expiry in years
            options_df['tte'] = options_df['expiration'].apply(
                lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d').date() - datetime.date.today()).days
            )
            
            # Calculate moneyness (K/S)
            options_df['moneyness'] = options_df['strike'] / self.spot_price
            
            # Calculate Greeks
            greeks = options_df.apply(self.calculate_greeks, axis=1)
            options_df = pd.concat([options_df, greeks], axis=1)
            
            return options_df
            
        except Exception as e:
            raise ValueError(f"Error processing options chain: {str(e)}")
    
    def calculate_implied_vol_surface(self):
        options_df = self.get_options_chain()
        
        # Create pivot table for the heatmap
        vol_surface = pd.pivot_table(
            options_df,
            values='impliedVolatility',
            index=pd.qcut(options_df['moneyness'], 10),  # Bucketed moneyness
            columns=pd.qcut(options_df['tte'], 10),      # Bucketed time to expiry
            aggfunc='mean'
        )
        
        return vol_surface
    
    def plot_volatility_surface(self):
        vol_surface = self.calculate_implied_vol_surface()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            vol_surface,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Implied Volatility'}
        )
        
        plt.title(f'Implied Volatility Surface for {self.ticker}')
        plt.xlabel('Time to Expiry Buckets')
        plt.ylabel('Moneyness (Strike/Spot) Buckets')
        plt.tight_layout()
        plt.show()
    
    def plot_greeks_surface(self, greek='delta'):
        """Plot surface for specified Greek"""
        options_df = self.get_options_chain()
        
        greek_surface = pd.pivot_table(
            options_df,
            values=greek,
            index=pd.qcut(options_df['moneyness'], 10),
            columns=pd.qcut(options_df['tte'], 10),
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            greek_surface,
            cmap='RdYlBu',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': greek.capitalize()}
        )
        
        plt.title(f'{greek.capitalize()} Surface for {self.ticker}')
        plt.xlabel('Time to Expiry Buckets')
        plt.ylabel('Moneyness (Strike/Spot) Buckets')
        plt.tight_layout()
        plt.show()
        
    def summarize_options(self):
        options_df = self.get_options_chain()
        
        print(f"\nOptions Analysis Summary for {self.ticker}")
        print(f"Current Stock Price: ${self.spot_price:.2f}")
        print(f"\nTotal number of options: {len(options_df)}")
        print(f"Number of expiration dates: {len(options_df['expiration'].unique())}")
        
        print("\nImplied Volatility Summary:")
        print(options_df['impliedVolatility'].describe())
        
        print("\nGreeks Summary (Mean Values):")
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            print(f"{greek.capitalize()}: {options_df[greek].mean():.4f}")
        
        # Analysis by moneyness
        print("\nAverage Implied Volatility by Moneyness:")
        moneyness_bins = pd.qcut(options_df['moneyness'], 5)
        print(options_df.groupby(moneyness_bins)['impliedVolatility'].mean())

def analyze_options(ticker: str):
    try:
        analyzer = OptionsAnalyzer(ticker)
        analyzer.summarize_options()
        analyzer.plot_volatility_surface()
        
        # Plot surfaces for each Greek
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            analyzer.plot_greeks_surface(greek)
            
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        raise

# Run analysis
if __name__ == "__main__":
    analyze_options("AAPL")  # Replace with your desired ticker

