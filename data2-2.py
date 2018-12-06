import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
for count,year in enumerate(range(100,90,-1)):
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
                            'ifrs': 'N'})
    fin = pd.read_html(read.text)[1]
    fin.index = fin.iloc[:,0]
    fin.columns = pd.concat([fin.iloc[0,:2] ,fin.iloc[1,:-2]]).tolist()
    fin.index = fin.iloc[:,0]
    fin = fin.drop("公司代號").copy()
    fin = fin.drop("負債佔資產比率(%)").copy()
    fin = fin.reset_index(drop=True)
    
    if count ==0:
        fout = fin.copy()
        len_stock = len(fin)
    fout = pd.concat([fout,fin],axis=1,join_axes=[fout.index]).copy()