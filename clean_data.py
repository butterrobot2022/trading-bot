import pandas as pd
from backtesting.test import GOOG, SMA, EURUSD


df = pd.read_csv('/Users/motin/Downloads/traffic/traffic/XAUUSD15.csv').drop_duplicates()
df.index = df['Time'].values
del df['Time']
print(df)



print(EURUSD)