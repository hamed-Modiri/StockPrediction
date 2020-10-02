import sys
import os.path

sys.path.append('.')
sys.path.append('./Infra')
sys.path.append('./Deep')


import Downloadfiles as mydf  
import myIndicators
import LSTMforecast

download=mydf.clsDownloadfiles()
#stocks=download.getallStocks()
stocks=download.getmyportfo()
#stocks=download.getselectedStocks()

i=0
for i in range(len(stocks)):
    if os.path.isfile("./Dataset/"+stocks[i]+'.csv'):
        myIndicators.ShowIndicators(stocks[i],stocks[i])
i=0
for i in range(len(stocks)):
    if os.path.isfile("./Dataset/"+stocks[i]+'.csv'):
        LSTMforecast.LSTMforecast(stocks[i],stocks[i],True,2,30)

print("end")