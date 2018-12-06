# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:21:58 2018

@author: StPioneer
"""

import math 
from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM 
from keras.callbacks import EarlyStopping

def report(strategy,close = "Close"):
    
    strategy = [["股東權益報酬率(%)",5,15],
                ["純益率(%)",5,10]]
    close= "Adj Close"   
    train_x = np.zeros((1,int(max(np.array(strategy)[:,1])),len(strategy)))#training資料存放區
    report = pd.DataFrame([[]])#達掉條件的股票代號及報酬率
    for root, dirs, files in walk("data/report/財務分析/"):print("讀取路徑路徑檔案")
    for file in files:            
    
        fin_data  = pd.read_csv("data/report/財務分析/"+file,index_col =0, parse_dates=True)
        try:
            fin_price = pd.read_csv("data/price/"+file,index_col =0, parse_dates=True)
        except:continue
        
        for i in range(1,len(fin_data)-1,1):
            br = 0
            for st in strategy:
                #財報檢查長度是否有問題
                if len(fin_data.iloc[i:i+st[1]]) !=st[1] :
                    br = 1
                    break
                try:
                    #檢查數據是否為空
                    [fin_data.iloc[i:i+st[1]][st[0]].mean(),
                     fin_data.iloc[i:i+st[1]].index[0].year,
                     fin_data.iloc[i:i+st[1]][st[0]].index[0],
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year)+"-04"][close].iloc[0],
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year)+"-04"][close].iloc[5],#印出價格隔年4月第一次開盤價格
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year)+"-05"][close].iloc[0],
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year)+"-07"][close].iloc[0],
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year)+"-10"][close].iloc[0],
                     fin_price[str(fin_data.iloc[i:i+st[1]][st[0]].index[0].year+1)+"-04"][close].iloc[0]]
                    #,fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+2)+"-04"]["Adj Close"].iloc[0]
                    #,fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+3)+"-04"]["Adj Close"].iloc[0]
                except: 
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
                                       
                
                train_x = np.vstack((train_x,train_x1.reshape(1,np.shape(train_x1)[0],np.shape(train_x1)[1]))).copy()
    
    report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y"]
    #report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y","evenPrice2y","evenPrice3y"]
    
    report= report[1:]
    train_x = train_x[1:]
    report.index = pd.to_datetime(report.iloc[:,0].copy())#Index轉乘時間序列
    return report,train_x

def reportMediam(report):
    #將資料以年為單位整理，並且轉化權益成報表
    for i in range(report.index.min().year,report.index.max().year+1,1):
        reportout = report[str(i)].copy()
        
        out=  reportout.iloc[:,0:2].copy()
        out = pd.concat([out,
                        reportout["evenPrice"],#事發當天價格
                        (reportout["evenPrice5d"]-reportout["evenPrice"])*100/reportout["evenPrice"],##後五天漲跌幅
                        (reportout["evenPrice1m"]-reportout["evenPrice"])*100/reportout["evenPrice"],
                        (reportout["evenPrice3m"]-reportout["evenPrice"])*100/reportout["evenPrice"],
                        (reportout["evenPrice6m"]-reportout["evenPrice"])*100/reportout["evenPrice"],
                        (reportout["evenPrice1y"]-reportout["evenPrice"])*100/reportout["evenPrice"]],axis=1)
                        #(reportout["evenPrice2y"]-reportout["evenPrice"])*100/reportout["evenPrice"],
                        #(reportout["evenPrice3y"]-reportout["evenPrice"])*100/reportout["evenPrice"]
        out.columns = report.columns.copy()
        
        reportY = out.describe()#儲存統計總攬
        reportY.iloc[0,1] = i#儲存年份於count第二列
        if i == report.index.min().year:
            reportYA=  np.array(reportY).reshape(1,np.shape(reportY)[0],np.shape(reportY)[1])#轉為array
            
            
        else: reportYA = np.vstack((reportYA,np.array(reportY).reshape(1,np.shape(reportY)[0],np.shape(reportY)[1])))#合併所有年份  
    return ((reportYA[:,5,5]/100)+1).cumprod() ,reportYA #中位數累乘
def normalize(fin):
    fin_out = fin.copy()
    if fin.ndim==3:        
        for i in range(0,np.shape(fin_out)[2]):
            #將nan填充為均值        
            fin_out[np.isnan(fin_out[:,:,i])]= np.mean(fin_out[~np.isnan(fin_out[:,:,i])][i])
            fin_out[:,:,i] = (fin_out[:,:,i]-fin_out[:,:,i].min())/(fin_out[:,:,i].max()-fin_out[:,:,i].min())
    if fin.ndim<=2: 
        for i in range(0,np.shape(fin_out)[1]):
            fin_out[:,i]=(fin_out[:,i]-fin_out[:,i].min())/(fin_out[:,i].max()-fin_out[:,i].min())
    return fin_out
def denorm(fin_o,fin):
    fin_out = fin.copy()
    if fin.ndim==3:
        for i in range(0,np.shape(fin_out)[2]):
            fin_out[np.isnan(fin_out)] = np.mean(fin_out[~np.isnan(fin_out[:,:,i])])
            fin_out[:,:,i] = fin[:,:,i]*(fin_o[:,:,i].max()-fin_o[:,:,i].min())+fin_o[:,:,i].min()
    if fin.ndim<=2:
        for i in range(0,np.shape(fin_out)[1]):
            fin_out[:,i] = fin[:,i]*((fin_o[:,i].max()-fin_o[:,i].min()))+fin_o[:,i].min()
    return fin_out
def build_model(input_length, input_dim,output_dim):
    d = 0.25
    model = Sequential()

    model.add(LSTM(64, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(32, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(6,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(output_dim,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

report ,x  = report([["股東權益報酬率(%)",15,-100],
                           ["純益率(%)",15,-100]],"Adj Close")
reportMediam,reportYA= reportMediam(report)

y = np.round(np.array((report["evenPrice1y"]-report["evenPrice"])*100/report["evenPrice"]).reshape(-1,1),2)

norm_x = normalize(x)
norm_y = normalize(y)

train_x = norm_x[:round(len(x)*0.9)]
train_y = norm_y[:round(len(y)*0.9)]

test_x = norm_x[round(len(x)*0.9):]
test_y = y[round(len(y)*0.9):]

model = build_model(np.shape(x)[1], np.shape(x)[2],np.shape(y)[1])
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
train_history = model.fit( train_x, train_y, batch_size=8, epochs=1000, validation_split=0.1, verbose=1)

pred = model.predict(test_x)
denorm_pred = denorm(y,pred)
denorm_close = denorm_pred[:,-1].reshape(-1,1)
plt.plot(denorm_close,color='red', label='Close_Prediction')
plt.plot(test_y[:,-1],color='blue', label='Close_Answer')
