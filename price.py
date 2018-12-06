# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:01:35 2018

@author: 重慶
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf

#--------------處理數據---------------#
fout = pd.read_excel('evaluate.xlsx',columns = [0])
#更改公司代碼
a = 0
count=0
ch = np.array(fout.iloc[:,1])
for i in ch:
    ch[a] = ch[a][0:4]
    a+=1
fout = fout.replace(np.array(fout.iloc[:,1]),ch)
#更改日期
fout["評等日期"]=pd.to_datetime(fout["評等日期"],format='%Y%m%d')

#現價>目標價
#an = fout[fout["現價"]>=fout["新目標價"]]
#現價>目標價5%
#an2 = fout[((fout["新目標價"]-fout["現價"])/fout["現價"])<-0.05] 
#降級
degrade = fout[fout["升/降"]=="降級"]
#degrade2 = fout[(fout["升/降"]=="降級")&(fout["現價"]>=fout["新目標價"])]
for eventday, nb in np.array(degrade.iloc[:200,0:2]):

    number = 3008
    event = eventday
    start_date = '2009-09-18'
    end_date = '2018-10-06'
    
    #存檔為不為0之數字
    save = 0
    
    try:data = yf.download(tickers = [number+'.TW'], start = start_date, end = end_date)
    except:
        try:
            data = yf.download(tickers = [number+'.TWO'], start = start_date, end = end_date)
        except:continue
        else:data = yf.download(tickers = [number+'.TWO'], start = start_date, end = end_date)
    else:data = yf.download(tickers = [number+'.TW'], start = start_date, end = end_date)
    
    data.head()
 """  
    p = 100*(np.array(data.Close[1:])-np.array(data.Close[:-1]))/np.array(data.Close[:-1])
    
    plt.plot(data.Close)
    plt.show()
    
    
    print("平均漲跌幅 : ",round(p.mean(),2))
    print("\n年期望值 : ",round(p.mean()*365,2))
    print("\n變異數 : ",round(p.std(),2))
    print("\n一倍標準差 : ",round(p.mean()+p.std(),2))
    print("\n兩倍標準差 : ",round(p.mean()+2*p.std(),2))
    
    epClose = np.array(data.loc[event:].Close)
    epOpen = np.array(data.loc[event:].Open)
    print("========================")
    if(len(epClose)>=2):print("\n事發隔日漲跌幅 ： ",round(100*(epClose[1]-epOpen[1])/epOpen[1],2))
    if(len(epClose)>=4):print("\n事發三日漲跌幅 ： ",round(100*(epClose[3]-epOpen[1])/epOpen[1],2))
    if(len(epClose)>=6):print("\n事發五日漲跌幅 ： ",round(100*(epClose[5]-epOpen[1])/epOpen[1],2))
    plt.plot(data.loc[event:].Open,"-o",color = 'red')
    plt.plot(data.loc[event:].Close,"-o")
    plt.show()
    
    f=pd.DataFrame(np.array([number,event,round(100*(epClose[1]-epOpen[1])/epOpen[1],2),round(100*(epClose[3]-epOpen[1])/epOpen[1],2),round(100*(epClose[5]-epOpen[1])/epOpen[1],2),round(p.mean(),2),round(p.mean()*365,2),round(p.std(),2),round(p.mean()+p.std(),2),round(p.mean()+2*p.std(),2)]).reshape(1,10),columns=['股號','事發日期','事發隔日漲跌幅','事發三日漲跌幅','事發五日漲跌幅','平均漲跌幅','年期望值','變異數','一倍標準差','兩倍標準差'])
    if count ==0:
        h =f.copy()
    h = h.append(f.copy()) 
    count+=1
    
    #存檔
    
    if(save):
        fin = pd.read_excel('journal.xlsx') 
        f.to_excel('journal.xlsx')
"""