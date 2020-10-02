
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
sns.set()

def ShowStocks(ori_name,ori_name_real,blnsave=True):
    # In[2]:
    matplotlib.rcParams['interactive'] == True
    modelname="meanvar"
    datasetname=""
    directory = './dataset/'
    #ori_name = ['IranConstInv', 'ReylPardazseir', 'S_ENBank', 'S_IranChemInd', 'S_RayanSaipa', 'TaminPetro']
    stocks = [directory + s +'.csv' for s in ori_name]

    # In[3]:

    #dfs = [pd.read_csv(s)[['Date', 'Close']] for s in stocks]
    #dfs = [pd.read_csv(s)[['<DTYYYYMMDD>', '<CLOSE>']] for s in stocks]
    dfs=[]
    for s in range(len(stocks)-1):
        df1 = pd.read_csv(stocks[s])
        df = pd.DataFrame(df1)
        df.columns= ['Ticker', 'Date', 'First','High','Low','Close','Value','Volume','OPENINT','Per','Open', 'Last' ]
        #del df['Ticker']
        df=df[['Date','Close']]
        df['Date']=pd.to_datetime(df['Date'], yearfirst=True,format='%Y%m%d')
        df=df.sort_values(by='Date',ascending=True)
        print(df.head(10))
        dfs.append(df)

    # In[4]:

    from functools import reduce
    data = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs).iloc[:, 1:]
    #data = reduce(lambda left,right: pd.merge(left,right,on='<DTYYYYMMDD>'), dfs).iloc[:, 1:]

    # In[5]:

    returns = data.pct_change()
    mean_daily_returns = returns.mean()
    volatilities = returns.std()


    # In[8]:

    #combine = pd.DataFrame({'returns': mean_daily_returns * 252,
    #                       'volatility': volatilities * 252})
    combine = pd.DataFrame({'returns': mean_daily_returns ,
                           'volatility': volatilities })

    # In[9]:

    g = sns.jointplot("volatility", "returns", data=combine, kind="reg",height=7)

    for i in range(combine.shape[0]):
        plt.annotate(ori_name[i].replace('.csv',''), (combine.iloc[i, 1], combine.iloc[i, 0]))
    
    plt.text(0, -1.5, 'SELL', fontsize=25)
    plt.text(0, 1.0, 'BUY', fontsize=25)
    
    plt.show()
    # In[ ]:'

    if blnsave:
        plt.savefig("./Results/"+modelname+"_"+datasetname+"_"+datetime.now().strftime('%Y-%m-%d%H%M%S') +".png")

if __name__ == "__main__":
    ori_name = ['28809886765682162','45392752356003555']
    ShowStocks(ori_name,ori_name,False)