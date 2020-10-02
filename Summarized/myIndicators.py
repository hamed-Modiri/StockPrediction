
# coding: utf-8

# In[1]:

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')


# In[2]:

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
import datetime as dt
from tqdm import tqdm
import statsmodels.api as sm

#from matplotlib.mpl_finance import candlestick_ohlc
from matplotlib.dates import DateFormatter
from scipy import stats

import string

def ShowIndicators(datasetname,stock_name,blnsave=True):

    sns.set()
    #tf.compat.v1.random.set_random_seed(1234)
    matplotlib.rcParams['interactive'] == True

    # Read Data

    modelname="Single"
    #stock_name="Glucosan"
    #datasetname="Glucosan"
    df = pd.read_csv('./dataset/'+ datasetname +'.csv')

    quotes = pd.DataFrame(df)
    quotes.columns= ['Ticker','Date', 'First','High','Low','Close','Value','Volume','OPENINT','Per','Open', 'Last' ]
    del quotes['Ticker']
    quotes['Date']=pd.to_datetime(quotes['Date'], yearfirst=True,format='%Y%m%d')
    quotes.set_index('Date',inplace = True, append = False, drop = True)
    quotes['PctChange']=((quotes['Close']-quotes['Open'])/quotes['Open'])*100
    quotes = quotes.iloc[::-1]  # reverse order
    #print(quotes.head(10))

    # Time window for the analysis (should be consistent with the provided data)
    date_start = dt.date(2018, 1, 1)
    #date_end = dt.date(2017, 3, 31)
    # If last day use this
    dateend = quotes.tail(1).index
    date_end = dateend.to_pydatetime()[0]
    # Simple log returns (%) of Close price
    quotes['log_returns'] = np.log(quotes['Close']).diff()*100

    #Compute Indecies #################

    # Simple Moving Averages
    sma20 = quotes['Close'].rolling(window=20).mean()
    sma50 = quotes['Close'].rolling(window=50).mean()
    sma200 = quotes['Close'].rolling(window=200).mean()

    # Disparity index
    # The disparity index (or disparity ratio), compares, as a percentage, 
    # the latest close price to a chosen moving average. SMA (20) is used.
    disparity = ((sma20-quotes['Close'])/sma20)*100.

    # Exponential Moving Averages
    ema20 = quotes['Close'].ewm(span=20).mean()
    ema50 = quotes['Close'].ewm(span=50).mean()
    ema200 = quotes['Close'].ewm(span=200).mean()

    # Bollinger bands
    ma20 = quotes['Close'].rolling(window=20).mean()
    sd20 = quotes['Close'].rolling(window=20).std()
    lowr = ma20 - 2*sd20
    uppr = ma20 + 2*sd20

    # Bollinger BandWidth (20)
    bb_width = ((uppr - lowr)/ma20)*100.
    bb_width_ma200 = bb_width.rolling(window=200).mean()

    # MACD indicator - MACD(12,26,9)
    #moving_average_convergence_divergence_macd
    ema12 = quotes['Close'].ewm(span=12).mean()
    ema26 = quotes['Close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    # Percentage Price Oscillator - PPO(12,26,9)
    ppo_line = ((ema12 - ema26)/ema26)*100.
    ppo_signal_line = ppo_line.ewm(span=9).mean()
    ppo_hist = ppo_line - ppo_signal_line

    # Percentage Volume Oscillator - PVO(12,26,9)
    vol_ema12 = quotes['Volume'].ewm(span=12).mean()
    vol_ema26 = quotes['Volume'].ewm(span=26).mean()
    pvo_line = ((vol_ema12 - vol_ema26)/vol_ema26)*100.
    pvo_signal_line = pvo_line.ewm(span=9).mean()
    pvo_hist = pvo_line - pvo_signal_line

    # Stochastic Oscillator (20,5,5)
    lowest_low = quotes['Low'].rolling(window=20).min()
    highest_high = quotes['High'].rolling(window=20).max()
    osc_line = ((quotes['Close'] - lowest_low)/(highest_high - lowest_low))*100.
    osc_line = osc_line.rolling(window=5).mean()  # SMA(5)
    osc_signal = osc_line.rolling(window=5).mean()  # SMA(5)

    # Compute RSI (14 days)
    def relative_strength(prices, n=14):
        deltas = np.diff(prices)
        seed = deltas[:n+1]
        up = seed[seed >= 0].sum()/n
        down = -seed[seed < 0].sum()/n
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1. + rs)
        for i in range(n, len(prices)):
            delta = deltas[i - 1]  # cause the diff is 1 shorter
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(n - 1) + upval)/n
            down = (down*(n - 1) + downval)/n
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
        return rsi
    rsi = relative_strength(quotes['Close'])


    # Candlesticks (data preparation)
    tindex = quotes.index.values
    ttime = []
    for k in tindex:
        y, m, d = str(k).split('-')
        ttime.append(dt.date(int(y), int(m), int(d[:2])))
    mtime = mdates.date2num(ttime)
    ohlc = quotes[['Open', 'High', 'Low', 'Close']].values
    qdata = np.c_[mtime, ohlc]


    # Simple log returns
    #returns = quotes['log_returns'].ix[date_start:date_end].dropna()
    returns = quotes['log_returns'].loc[date_start:date_end].dropna()

    # Daily VaR and CVaR from historical time window at 95% confidence
    VaR = stats.scoreatpercentile(returns.values, 5)
    ind = returns.values < VaR
    CVaR = np.mean(returns[ind])
    print('VaR = {:.2f} %, CVaR = {:.2f} %'.format(VaR, CVaR))

    # Mean and standard deviation of the simple log returns
    mu_logret = returns.mean()
    sd_logret = returns.std()
    print('Mean return: {:.2f} %'.format(mu_logret))
    print('St.dev. return: {:.2f}'.format(sd_logret))


    # Volume-by-Price indicator
    # Volume-by-Price calculations are based on the entire period displayed on the chart. 
    # Volume-by-Price calculations do not extend beyond the historical data shown on the chart.
    # This example is based on closing prices and the default parameter setting (12):
    #    1. Find the high-low range for closing prices for the entire period.  
    #    2. Divide this range by 12 to create 12 equal price zones.
    #    3. Total the amount of volume traded within each price zone.  

    #promet = quotes['Promet'].ix[date_start:date_end]
    #prices = quotes['Close'].ix[date_start:date_end]
    #_, zones = np.histogram(prices, bins=12)
    #vol_by_price = []
    #for i in range(len(zones)-1):
    #    vol_by_price.append(promet[(prices > zones[i]) & (prices <= zones[i+1])].sum())
    #vol_by_price = np.asarray(vol_by_price)*1e-6  # Milions (kn)


    # Ichimoku Cloud plot
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
    high9 = quotes['High'].rolling(window=9).max()
    low9 = quotes['Low'].rolling(window=9).min()
    conversion_line = (high9 + low9)/2.
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
    high26 = quotes['High'].rolling(window=26).max()
    low26 = quotes['Low'].rolling(window=26).min()
    base_line = (high26 + low26)/2.
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
    # This value is plotted 26 periods in the future
    leading_span_A = ((conversion_line + base_line)/2.).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
    # This value is plotted 26 periods in the future
    high52 = quotes['High'].rolling(window=52).max()
    low52 = quotes['Low'].rolling(window=52).min()
    leading_span_B = ((high52 + low52)/2.).shift(26)
    # Chikou Span (Lagging Span): Close plotted 26 days in the past
    lagging_span = quotes['Close'].shift(-26)

    # Round number to the nearest base
    def myround(x, base=5):
        return int(base * round(float(x)/base))







    #Plot 
    fig = plt.figure(figsize=(12,8.5))
    gx = gs.GridSpec(nrows=5, ncols=2, height_ratios=[1,2,1,1,1], width_ratios=[5,1])
    axt = fig.add_subplot(gx[0,0])
    ax0 = fig.add_subplot(gx[1,0], sharex=axt)  # main 
    axr = ax0.twinx()
    ax1 = fig.add_subplot(gx[2,0], sharex=ax0)
    axq = fig.add_subplot(gx[0,1])
    axh = fig.add_subplot(gx[1,1])
    axv = fig.add_subplot(gx[2,1])
    ax3 = fig.add_subplot(gx[3,0], sharex=ax0)
    ax4 = fig.add_subplot(gx[4,0], sharex=ax0)

    # Top figure (left)
    axt.set_title(stock_name)
    axt.plot(quotes.index, osc_line, color='darkorange', lw=2, label='Stochastic Oscillator (20,5,5)')
    axt.plot(quotes.index, osc_signal, color='seagreen', lw=1.5, label='Signal line')
    axt.axhline(80, color='grey')
    axt.axhline(50, color='grey', ls='--')
    axt.axhline(20, color='grey')
    axt.fill_between(quotes.index, osc_line, 80, where=(osc_line >= 80), 
                     interpolate=True, color='darkorange', alpha=0.8)
    axt.fill_between(quotes.index, osc_line, 20, where=(osc_line <= 20), 
                     interpolate=True, color='darkorange', alpha=0.8)
    axt.text(0.6, 0.9, '>80 = overbought', va='top', transform=axt.transAxes, fontsize=12)
    axt.text(0.6, 0.1, '<20 = oversold', transform=axt.transAxes, fontsize=12)
    #axt.set_ylim(0, 100)
    axt.set_yticks([20, 80])
    axt.legend(loc='lower left', frameon='fancy', fancybox=True, framealpha=0.5)
    # Middle figure left
    # Volume (mountain plot or bars plot)
    vmax = quotes['Volume'].loc[date_start:date_end].max()
    if len(quotes['Volume'].loc[date_start:date_end]) > 250:
        axr.fill_between(quotes.index, quotes[u'Volume'], 0, color='darkorange', alpha=0.8)
        axr.set_yticks([])
        axr.grid(False)
    else:
        axr.bar(quotes.index, quotes[u'Volume'], width=0.8, color='darkorange', alpha=0.8)
        axr.set_yticks([])
        axr.grid(False)
    axr.set_ylim(0, 2*vmax)
    # EMA (20), EMA (50) and EMA (200)
    ax0.plot(quotes.index, ema20, ls='-', lw=1.5, c='magenta', label='EMA (20)')
    ax0.plot(quotes.index, ema50, ls='-', lw=1.5, c='royalblue', label='EMA (50)')
    ax0.plot(quotes.index, ema200, ls='-', lw=1.5, c='seagreen', label='EMA (200)')
    # Bollinger band (20)
    ax0.fill_between(quotes.index, lowr, uppr, color='wheat', alpha=0.5, label='BB (20)')
    ax0.plot(quotes.index, lowr, ls='-', lw=0.5, c='tan', label='')
    ax0.plot(quotes.index, uppr, ls='-', lw=0.5, c='tan', label='')
    # Candlestics
    if len(quotes['Close'].loc[date_start:date_end]) > 250:
        candle_width = 1
    else:
        candle_width = 0.8  # default
    #candlestick_ohlc(ax0, qdata, width=candle_width)  # candlesticks
    ax0.legend(loc='upper left', frameon='fancy', fancybox=True, shadow=True, framealpha=0.5)
    # Print OHLC for the last trading day
    last = quotes.iloc[-1]
    s = '{:s}: O:{:g}, H:{:g}, L:{:g}, C:{:g}, \
    Chg:{:g}%'.format(str(last.name)[:10], last['Open'], last[u'High'], last[u'Low'], 
                      last['Close'], last['PctChange'])
    ax0.text(0.3, 0.9, s, transform=ax0.transAxes, fontsize=12, fontweight='bold')
    ymin = min(quotes['Low'].loc[date_start:date_end].min(), 
               lowr.loc[date_start:date_end].min(), 
               ema20.loc[date_start:date_end].min(), 
               ema50.loc[date_start:date_end].min(), 
               ema200.loc[date_start:date_end].min())
    ymax = max(quotes['High'].loc[date_start:date_end].max(), 
               uppr.loc[date_start:date_end].max(), 
               ema20.loc[date_start:date_end].max(), 
               ema50.loc[date_start:date_end].max(), 
               ema200.loc[date_start:date_end].max())
    diff = (ymax - ymin)*0.05
    ax0.set_ylim(myround(ymin-diff), myround(ymax+diff))
    # Bottom figure (left)
    # Percenage Volume Oscilator (PVO)
    ax1.plot(quotes.index, pvo_line, ls='-', lw=1.5, c='royalblue', label='PVO line')
    ax1.plot(quotes.index, pvo_signal_line, ls='-', lw=1, c='seagreen', label='Signal line')
    if len(pvo_line.loc[date_start:date_end]) > 250:
        ax1.bar(quotes.index, pvo_hist, width=1, color='red')
    else:
        ax1.bar(quotes.index, pvo_hist, width=0.8, color='red')
    ymin_pvo = min(pvo_line.loc[date_start:date_end].min(),
                   pvo_signal_line.loc[date_start:date_end].min(),
                   pvo_hist.loc[date_start:date_end].min())
    ymax_pvo = max(pvo_line.loc[date_start:date_end].max(),
                   pvo_signal_line.loc[date_start:date_end].max(),
                   pvo_hist.loc[date_start:date_end].max())
    diff_pvo = (ymax_pvo - ymin_pvo)*0.1
    ax1.set_ylim(int(ymin_pvo-diff_pvo), int(ymax_pvo+diff_pvo))
    ax1.text(0.02, 0.1, 'PVO (12,26,9)', transform=ax1.transAxes, fontsize=12, 
             fontweight='bold', color='red', backgroundcolor='white')
    ax1.legend(loc='upper left', frameon='fancy', fancybox=True, framealpha=0.5)
    ax1.set_xlim(date_start, date_end)  # Clipping view
    # Right column figures deal with simple log returns analysis
    # Top right figure
    ret_data = returns.values
    #axq.set_title('Normality test', fontsize=12)
    # Q-Q plot of simple log returns (Normality test)
    sm.graphics.qqplot(ret_data, stats.norm, fit=True, line='q', ax=axq)
    axq.text(0.1, 0.8, 'QQ plot', transform=axq.transAxes, fontsize=12, 
             color='blue', backgroundcolor='white')
    axq.set_xlabel('')
    axq.set_ylabel('')
    # Middle right figure
    #axh.set_title('Daily at 95%', fontsize=11)
    lower_var = ret_data[ind]
    upper_var = ret_data[~ind]
    upper_var = upper_var[~np.isnan(upper_var)]
    # Number of bins is fixed here, but can be determined using 
    # any of the methods availabale in the numpy.hist function!
    axh.hist(upper_var, bins=20, orientation='horizontal', color='skyblue', label='')
    axh.hist(lower_var, bins=10, orientation='horizontal', color='red', label='')
    axh.text(0.1, 0.8, 'Daily (95% conf.)\nVaR = {:.2f}%\nCVaR = {:.2f}%'.format(abs(VaR), abs(CVaR)), 
             transform=axh.transAxes, fontsize=12, backgroundcolor='white')
    # Bottom right figure (violin plot)
    sns.violinplot(ret_data, orient='v', saturation=0.5, ax=axv)
    axv.set_ylim(-5, 5)
    axv.set_xlabel('Log returns', fontsize=12)
    plt.tight_layout()

    # rsi figure
    ax3.plot(quotes.index, rsi, color='darkorange', label='Relative strength index - RSI (14)')
    ax3.axhline(70, color='grey')
    ax3.axhline(50, color='grey', ls='--')
    ax3.axhline(30, color='grey')
    ax3.fill_between(quotes.index, rsi, 70, where=(rsi >= 70), interpolate=True, color='darkorange', alpha=0.8)
    ax3.fill_between(quotes.index, rsi, 30, where=(rsi <= 30), interpolate=True, color='darkorange', alpha=0.8)
    ax3.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax3.transAxes, fontsize=12)
    ax3.text(0.6, 0.1, '<30 = oversold', transform=ax3.transAxes, fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.set_yticks([30, 70])
    ax3.legend(loc='lower left')

    # MACD figure
    ax4.plot(quotes.index, macd_line, ls='-', lw=1.5, c='royalblue', label='MACD line')
    ax4.plot(quotes.index, signal_line, ls='-', lw=1, c='seagreen', label='Signal line')
    if len(macd_line.loc[date_start:date_end]) > 250:
        ax4.bar(quotes.index, macd_hist, width=1, color='red')
    else:
        ax4.bar(quotes.index, macd_hist, width=0.8, color='red')
    ymin_macd = min(macd_line.loc[date_start:date_end].min(),
                    signal_line.loc[date_start:date_end].min(),
                    macd_hist.loc[date_start:date_end].min())
    ymax_macd = max(macd_line.loc[date_start:date_end].max(),
                    signal_line.loc[date_start:date_end].max(),
                    macd_hist.loc[date_start:date_end].max())
    diff_macd = (ymax_macd - ymin_macd)*0.1
    ax4.set_ylim(int(ymin_macd-diff_macd), int(ymax_macd+diff_macd))
    ax4.text(0.02, 0.1, 'MACD (12,26,9)', transform=ax4.transAxes, fontsize=12, 
             fontweight='bold', color='red', backgroundcolor='white')
    ax4.legend(loc='upper left', frameon='fancy', fancybox=True, framealpha=0.5)
    ax4.set_xlim(date_start, date_end)  # Clipping view


    plt.show()
    if blnsave:
        plt.savefig("./Results/"+modelname+"_"+datasetname+"_"+datetime.now().strftime('%Y-%m-%d%H%M%S') +".png", dpi=600)




if __name__ == "__main__":
    ShowIndicators("Glucosan","Glucosan",False)

