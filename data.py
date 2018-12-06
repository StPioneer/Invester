# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:53:49 2018

@author: user
"""

import requests
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
count =0
for year in range(2018,2012,-1):
    type ='營益分析彙總表'
    if year >= 1000:
        year -= 1911
    
    if type == '綜合損益彙總表':
        url = 'http://mops.twse.com.tw/mops/web/ajax_t163sb04'
    elif type == '資產負債彙總表':
        url = 'http://mops.twse.com.tw/mops/web/ajax_t163sb05'
    elif type == '營益分析彙總表':
        url = 'http://mops.twse.com.tw/mops/web/ajax_t163sb06'
    else:
        print('type does not match')
    r = requests.post(url, {'encodeURIComponent':"1",
            'step':"1",
            'firstin':"1",
            'off':"1",
            'TYPEK':'sii',
            'year':str(year),
            'season':"1"})
        
    r.encoding = 'utf8'
    dfs = pd.read_html(r.text)
        
        
    for i, df in enumerate(dfs):
        df.columns = df.iloc[0]
        dfs[i] = df.iloc[1:]
          
    df = pd.concat(dfs).applymap(lambda x: x if x != '--' else np.nan)
    """
    df.index= df["公司代號"]
    df.pop("公司代號")
    df.pop(df.columns[1])
    df.pop(df.columns[0])
    df = df[:-3].copy()
    df = df[df.index!="公司代號"].copy()
    if count ==0:
        fout = df.copy()
    count+=1
    fout = pd.concat([fout,df],axis=1,join_axes=[fout.index]).copy()

fout2 =np.array(fout)
fout2 = fout2.reshape(883,7,5).copy()
o = 0
for i in fout.index:
    h = pd.DataFrame(fout2[o],columns = df.columns)
    h.to_csv("data/report/"+fout.index[o]+'.csv',encoding='utf_8_sig')
    o+=1
    
    #df=df["公司代號"].copy()

    for nb in np.array(df):
        number = nb
        start_date = '2000-01-01'
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
        p = 100*(np.array(data.Close[1:])-np.array(data.Close[:-1]))/np.array(data.Close[:-1])
        p = p.round(decimals=2).copy()
        p = np.insert(p,0,values=0,axis=0)
        p = pd.DataFrame(p,columns = ["change"],index = data.index )
        data = pd.concat([data,p],axis=1).copy()
        data.to_csv("data/price/"+nb+'.csv')
"""