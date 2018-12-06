# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:35:25 2018

@author: StPioneer
"""

import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
start_date = '1999-01-01'
end_date = '2018-10-06'
number ="^TWII" 
#存檔為不為0之數字
save = 0

#data = yf.download(tickers = [number+'.TW'], start = start_date, end = end_date)
data = yf.download(tickers = [number], start = start_date, end = end_date)
data.head()
p = 100*(np.array(data.Close[1:])-np.array(data.Close[:-1]))/np.array(data.Close[:-1])
p = p.round(decimals=2).copy()
p = np.insert(p,0,values=0,axis=0)
p = pd.DataFrame(p,columns = ["change"],index = data.index )
data = pd.concat([data,p],axis=1).copy()
data.to_csv('data/price/TWII.csv')

