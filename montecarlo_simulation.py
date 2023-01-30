# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style # Allows us to choose the chart's color
from scipy.stats import norm

yf.pdr_override()
style.use('seaborn')

# Then we have to download the data

ticker = 'AAPL'
startdate = '2012-01-01'
enddate = '2023-01-30'
data = pd.DataFrame()

data[ticker] = pdr.get_data_yahoo(ticker, start=startdate, end=enddate)['Adj Close']

# Values

log_returns = np.log(1+data.pct_change())
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var) # Growth or decrease
std = log_returns.std() # Standard deviation due to we are working with a population and not with a sample

# Variables

days = 100
trials = 10000

# Now we need to add to our model the random component

Z = norm.ppf(np.random.rand(days,trials)) # We generate random values for a defined array
daily_returns = np.exp(drift.values + std.values * Z) # We generate the daily returns for each simulation based on a normal distribution
price_path = np.zeros_like(daily_returns) # We build the matrix that will allow us to make the multiplications
price_path[0] = data.iloc[-1] # We fix the day for the calculus

for t in range(1, days):
    price_path[t] = price_path[t-1]*daily_returns[t]

# We save this into a csv

np.savetxt('price_path.csv', price_path[t])

# Finally, we can graph these results

plt.figure(figsize=(15,6))

# Graph 1

plt.plot(pd.DataFrame(price_path))
plt.xlabel('Number of days')
plt.ylabel(ticker + ' Price')

#Graph 2

sns.displot(pd.DataFrame(price_path).iloc[-1])
plt.xlabel('Price of ' + str(days) + ' days')
plt.ylabel('Frequency')

plt.show()