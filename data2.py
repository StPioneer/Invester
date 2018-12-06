# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:57:23 2018

@author: user
"""

import requests
import pandas as pd
import numpy as np
import time

#http://mops.twse.com.tw/mops/web/t51sb02_q1
ifrs= lambda year:"Y"if year>101 else "N"
for count,year in enumerate(range(106,80,-1)):
    if year ==95:time.sleep( 60 )
    if year ==88:time.sleep( 60 )
    if year >= 1000:
        year -= 1911
    
    
    url = 'http://mops.twse.com.tw/mops/web/ajax_t51sb02'
    read = requests.post(url,data = {'encodeURIComponent':'1',
                            'run':'Y',
                            'step':'1',
                            'TYPEK':'sii',
                            'year':str(year),
                            'isnew':'',
                            'firstin':'1',
                            'off':'1',
                            'ifrs': ifrs(year)})
    fin = pd.read_html(read.text)[1]
    fin.index = fin.iloc[:,0]
    fin.columns = pd.concat([fin.iloc[0,:2] ,fin.iloc[1,:-2]]).tolist()
    fin.index = fin.iloc[:,0]
    fin = fin.drop("公司代號").copy()
    fin = fin.drop("負債佔資產比率(%)").copy()
    if year<=101:
        fin.pop(fin.columns[16])
        
    if count ==0:
        fout = fin.copy()
        len_stock = len(fin)
    fout = pd.concat([fout,fin],axis=1,join_axes=[fout.index]).copy()


fout2 = np.array(fout.iloc[:,21:])
fout2 = fout2.reshape(len_stock,(count+1),round(len(fout.iloc[0,:])/(count+2))).copy()

for count2,i in enumerate(fout.index):
    h = pd.DataFrame(fout2[count2],columns = fin.columns)
    if year <= 1000:
        year += 1911
    year +=count
    h = h[pd.isnull(h["公司代號"])!=True]
    h.index = np.arange(2017,2017-len(h),-1)[:len(h[pd.isnull(h["公司代號"])!=True])]
    h= h.iloc[:,2:].copy()
    h.to_csv("data/report/財務分析/"+fout.iloc[count2,0]+'.csv',encoding='utf_8_sig')
