import math 
from os import walk
import pandas as pd
import numpy as np
a = 0
strategy = [["股東權益報酬率(%)",15,15],["純益率(%)",15,10]]
close = "Adj Close"

train_x = np.zeros((1,int(max(np.array(strategy)[:,1])),len(strategy)+1))#training資料存放區
report = pd.DataFrame([[]])#達掉條件的股票代號及報酬率
for root, dirs, files in walk("data/price-report"):print("讀取路徑路徑檔案")
for file in files:            

    fin_data  = pd.read_csv("data/report/財務分析/"+file,index_col =0, parse_dates=True)
    fin_price = pd.read_csv("data/price/"+file,index_col =0, parse_dates=True)

    for i in range(1,4,1):
        br = 0
        for st in strategy:
            #財報檢查長度是否有問題
            if len(fin_data.iloc[i:i+st[1]]) !=st[1] :
                br = 1
                break

        if br ==1 :break 

        if(fin_data.iloc[i:i+st[1]].index[0].year<2000):break #超過2000年跳出迴圈(價格數據不夠)

        
        AND = True
        #設定條件
        for j in range(0,len(strategy),1):
            AND = AND and fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].mean() >= strategy[j][2]
        
        if (AND) : #條件達到   
            for count,k in enumerate(strategy):        
                if count ==0 : train_x1 =np.array(fin_data.iloc[i:i+k[1]][k[0]]).reshape(-1,1).copy()
                else :train_x1 = np.hstack((train_x1,np.array(fin_data.iloc[i:i+k[1]][k[0]]).reshape(-1,1).copy()))
            
            report = report.append(pd.DataFrame([[fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0],#事件輸出時間
                                   str(file[:4]),
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-04"][close].iloc[0], #印出價格隔年4月第一次開盤價格
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-04"][close].iloc[5], #印出價格隔年4月第一次開盤後5天價格
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-05"][close].iloc[0],
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-07"][close].iloc[0],
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-10"][close].iloc[0],
                                   fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year+1)+"-04"][close].iloc[0]]]))
                                   #fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+2)+"-04"]["Adj Close"].iloc[0],
                                   #fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+3)+"-04"]["Adj Close"].iloc[0]
                                   
            fin_price_report  = pd.read_csv("data/price-report/"+file,index_col =0, parse_dates=True)
            train_x1 = np.hstack((train_x1,np.array(fin_price_report.iloc[i:i+st[1]]["evenPrice"]).reshape(-1,1)))
            train_x = np.vstack((train_x,train_x1.reshape(1,np.shape(train_x1)[0],np.shape(train_x1)[1]))).copy()
            

report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y"]
#report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y","evenPrice2y","evenPrice3y"]

report= report[1:].copy()
train_x = train_x[1:].copy()
report.index = pd.to_datetime(report.iloc[:,0].copy())#Index轉乘時間序列
