# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:49:53 2018

@author: StPioneer
"""

from os import walk
import pandas as pd
import numpy as np


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

def TWII(yearmin,yearmax):#report.index.min().year
    data  = pd.read_csv("data/price/TWII.csv",index_col =0, parse_dates=True)
    TWII =[]
    for i in range(yearmin,yearmax+1+1,1):
        TWII.append(data[(data.index.month==4)&(data.index.year==i)].iloc[0]["Adj Close"])
    
    TWII  = np.array(TWII)    
    TWII = (TWII/TWII[0])
    return TWII

def report(strategy,close = "Close"):
    report = pd.DataFrame([[]])
    global files,fin_data,fin_price,file
    for root, dirs, files in walk("data/report/財務分析/"):print("讀取路徑路徑檔案")
    for file in files:            
    
        fin_data  = pd.read_csv("data/report/財務分析/"+file,index_col =0, parse_dates=True)
        try:
            fin_price = pd.read_csv("data/price/"+file,index_col =0, parse_dates=True)
        except:continue
        
        for i in range(1,len(fin_data)-1,1):
            br = 0
            
            for st in strategy:
                try:
                    [fin_data.iloc[i:i+st[1]].index[0].year,
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
                report = report.append( 
                                       pd.DataFrame([[fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0],#事件輸出時間
                                       str(file[:4]),
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-04"][close].iloc[0], #印出價格隔年4月第一次開盤價格
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-04"][close].iloc[5], #印出價格隔年4月第一次開盤後5天價格
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-05"][close].iloc[0],
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-07"][close].iloc[0],
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year)+"-10"][close].iloc[0],
                                       fin_price[str(fin_data.iloc[i:i+strategy[j][1]][strategy[j][0]].index[0].year+1)+"-04"][close].iloc[0]]]))
                                       #fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+2)+"-04"]["Adj Close"].iloc[0],
                                       #fin_price[str(fin_data.iloc[i:i+5]["股東權益報酬率(%)"].index[0].year+3)+"-04"]["Adj Close"].iloc[0]
    
        
    report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y"]
    #report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y","evenPrice2y","evenPrice3y"]
    
    report= report[1:]
    report.index = pd.to_datetime(report.iloc[:,0].copy())#Index轉乘時間序列
    return report


