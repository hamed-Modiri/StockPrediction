import pandas as pd

data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':['20200205', '20190205', '20200204', '20200305']}
data = pd.DataFrame(data)
print(data)
data.columns = ['Team Name', 'agg'] 
data['agg']=pd.to_datetime(data['agg'], yearfirst=True,format='%Y%m%d')
print(data)

df = pd.read_csv('./dataset/'+ datasetname +'.csv',usecols=[2,3,4,5,6,7,8,9,10,11,12])
del quotes['Vrsta prometa']

df.fillna(0)
df.dropna()