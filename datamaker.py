# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:14:44 2018

@author: user

將2000~2018股價整理成財報發布後5天、1月、3月、6月、1年表現

"""

from os import walk
import pandas as pd
import numpy as np


close = "Adj Close"
for root, dirs, files in walk("data/report/財務分析/"):print("讀取路徑路徑檔案")
for file in files: 
    br = 0
    fin_data  = pd.read_csv("data/report/財務分析/"+file,index_col =0, parse_dates=True)
    try:
        fin_price = pd.read_csv("data/price/"+file,index_col =0, parse_dates=True)

    except:continue
    report = pd.DataFrame([[]])
    for c,i in enumerate(fin_data):

        if(len(fin_data.index.year)<18):
            br=2
            break
        
        try:report = report.append(pd.DataFrame([[fin_data.index[c],#事件輸出時間
                           str(file[:4]),
                           round(fin_price[str(fin_data.index[c].year)+"-04"][close].iloc[0],2), #印出價格隔年4月第一次開盤價格
                           round(fin_price[str(fin_data.index[c].year)+"-04"][close].iloc[5],2), #印出價格隔年4月第一次開盤後5天價格
                           round(fin_price[str(fin_data.index[c].year)+"-05"][close].iloc[0],2),
                           round(fin_price[str(fin_data.index[c].year)+"-07"][close].iloc[0],2),
                           round(fin_price[str(fin_data.index[c].year)+"-10"][close].iloc[0],2),
                           round(fin_price[str(fin_data.index[c].year+1)+"-04"][close].iloc[0],2)]]))
        except:
            br=2
            break
        
        if(fin_data.index[c].year==2000):
            br=1
            break
    if br==2 :continue
    report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y"]
    #report.columns= ["eventime","index","evenPrice","evenPrice5d","evenPrice1m","evenPrice3m","evenPrice6m","evenPrice1y","evenPrice2y","evenPrice3y"]
    report= report[1:]
    report.index = pd.to_datetime(report.iloc[:,0])#Index轉乘時間序列
    report.pop("eventime")

    report.to_csv("data/price-report/"+file,encoding="utf_8_sig")
    if br==1 :continue