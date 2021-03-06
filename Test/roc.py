"""This is a example adding volume features.
"""
import pandas as pd
import ta

# Load data
df = pd.read_csv('./Dataset/S_ENBank.csv', sep=',')

# Clean nan values
df = ta.utils.dropna(df)

ta.momentum.roc(close=df['Close'])
print("End")
