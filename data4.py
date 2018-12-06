from os import walk
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM 
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
def data_helper():
    for root, dirs, files in walk("data/report/財務分析/"):print("讀取路徑路徑檔案")
    for file in files:
        fin_data  = pd.read_csv("data/report/財務分析/"+file,index_col =0, parse_dates=True)
        try:
            fin_price = pd.read_csv("data/price/"+file,index_col =0, parse_dates=True)
            fin_price["2017-04"].iloc[-1][3]
            fin_price["2016-04"].iloc[0][0]
            [fin_price["2016-04"].iloc[0][0],fin_price["2015-04"].iloc[0][0],
             fin_price["2014-04"].iloc[0][0],fin_price["2013-04"].iloc[0][0],fin_price["2012-04"].iloc[0][0],
             fin_price["2011-04"].iloc[0][0],fin_price["2010-04"].iloc[0][0]]
            fin_data.iloc[8]
        except:continue
        
        if file==files[0]:
            change =  np.hstack((np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2016']),np.array([fin_price["2016-04"].iloc[0][0]]).reshape(1,1)))
            #change = np.array((fin_price["2017-09"].iloc[-1][3]-fin_price["2017-04"].iloc[0][0])/\
            #                   fin_price["2017-04"].iloc[0][0]).reshape(1,1)
            
            #report2 = np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2016':'2010'])
            report2 = np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2015':'2010'])
            report =np.hstack((report2,np.array([fin_price["2015-04"].iloc[0][0],
                                                fin_price["2014-04"].iloc[0][0],
                                                fin_price["2013-04"].iloc[0][0],
                                                fin_price["2012-04"].iloc[0][0],
                                                fin_price["2011-04"].iloc[0][0],
                                                fin_price["2010-04"].iloc[0][0]]).reshape(6,1)))
            
            report = report.reshape(1,len(report),len(report[0]))
            
        else:
            change = np.vstack((change, np.hstack((np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2016']),np.array([fin_price["2016-04"].iloc[0][0]]).reshape(1,1)))))
            #change = np.vstack((change,np.array((fin_price["2017-09"].iloc[-1][3]-fin_price["2017-04"].iloc[0][0])/\
            #                                    fin_price["2017-04"].iloc[0][0]).reshape(1,1)))
            report2 = np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2015':'2010'])
            #report2 = np.array(fin_data[["負債佔資產比率(%)","資產報酬率(%)","股東權益報酬率(%)","每股盈餘(元)"]]['2016':'2010'])
            report2 = np.hstack((report2,np.array([fin_price["2015-04"].iloc[0][0],
                                                   fin_price["2014-04"].iloc[0][0],
                                                   fin_price["2013-04"].iloc[0][0],
                                                   fin_price["2012-04"].iloc[0][0],
                                                   fin_price["2011-04"].iloc[0][0],
                                                   fin_price["2010-04"].iloc[0][0]]).reshape(6,1)))
            
            report = np.vstack((report, report2.reshape(1,len(report2),len(report2[0]))))

    return np.round(np.float64(report.copy()),decimals=2) , np.round(np.float64(change.copy()),decimals=2)


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

x,y =data_helper()
norm_x = normalize(x)
norm_y = normalize(y)
#定義訓練集測試集
train_x = norm_x[:round(len(x)*0.9)]
train_y = norm_y[:round(len(y)*0.9)]

test_x = norm_x[round(len(x)*0.9):]
test_y = y[round(len(y)*0.9):]
#建立模型
model = build_model(np.shape(x)[1], np.shape(x)[2],np.shape(y)[1])
#訓練模型
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
train_history = model.fit( train_x, train_y, batch_size=8, epochs=50, validation_split=0.1, verbose=1)
pred = model.predict(test_x)
denorm_pred = denorm(y,pred)

denorm_close = denorm_pred[:,-1].reshape(-1,1)


plt.figure()
plt.plot(denorm_close,color='red', label='Close_Prediction')
plt.plot(test_y[:,-1],color='blue', label='Close_Answer')
plt.show()
compare = np.hstack([test_y[:,-1].reshape(-1,1),denorm_close,x[:,0,len(x[0,0])-1][round(len(x)*0.9):].reshape(len(denorm_pred),1)])

print("做多正確數量 : ",len(compare[np.logical_and(compare[:,1]>compare[:,2],compare[:,0]>compare[:,2])]))
print("做空正確數量 : ",len(compare[np.logical_and(compare[:,1]<compare[:,2],compare[:,0]<compare[:,2])]))
pl = compare[np.logical_and(compare[:,1]>compare[:,2],compare[:,0]>compare[:,2])]
ps = compare[np.logical_and(compare[:,1]<compare[:,2],compare[:,0]<compare[:,2])]
print("做多")
print("獲利次數 : ",len(pl))
print("總獲利 : ", np.sum((pl[:,0]-pl[:,2])*100/pl[:,2]))
ll = compare[np.logical_and(compare[:,1]>compare[:,2],compare[:,0]<compare[:,2])]
ls = compare[np.logical_and(compare[:,1]<compare[:,2],compare[:,0]>compare[:,2])]
print("損失次數 : ",len(ll))
print("總損失 : ", np.sum((ll[:,0]-ll[:,2])*100/ll[:,2]))

print("\n做空")
print("獲利次數 : ",len(ps))
print("總獲利 : ", np.sum(-(ps[:,0]-ps[:,2])*100/ps[:,2]))
print("損失次數 : ",len(ls))
print("總損失 : ", np.sum(-(ls[:,0]-ls[:,2])*100/ls[:,2]))

