# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:19:11 2019


"""

import pandas as pd
import numpy as np

data = pd.read_csv("odev_tenis.csv")

outlook = data.iloc[:,0:1].values
tem_hum = data.iloc[:,1:3].values
windly = data.iloc[:,3:4].values
play = data.iloc[:,-1:].values

print(outlook)
print(tem_hum)
print(windly)
print(play)

# Kategorik -> Numerik verilere convert

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features="all")

outlook = ohe.fit_transform(outlook).toarray()
windly = ohe.fit_transform(windly).toarray()
play = ohe.fit_transform(play).toarray()

print(outlook,windly,play)

# Numeric verilerin tumunu df e cevirelim
s1 = pd.DataFrame(data =outlook, index = range(14), columns = ['outlook1','outlook2','outlook3'])
print(s1)
s2 = pd.DataFrame(data =tem_hum, index = range(14), columns = ['temperature','humidity'])
print(s2)
s3 = pd.DataFrame(data =windly[:,-1:], index = range(14), columns = ['windly']) # Dummy variable engellendi
print(s3)
plays = pd.DataFrame(data =play[:,-1:], index = range(14), columns = ['play'])   # Dummy variable engellendi
print(plays)
# Donusturulen numeric varileri birlestirelim

datas = pd.concat([s1,s2,s3],axis = 1)  #   outlook(1,2,3), temperature, humidity, windly
print(datas)

# training icin parcalama yapalim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(datas, plays, test_size = 0.33, random_state= 0 )

# training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

print(y_predict)

# backward elemination icin 1 kolonu ekleyelim 
import statsmodels.formula.api as sm
one_column = np.append(arr = np.ones((14,1)).astype(int),values = datas, axis = 1)

X_list = datas.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog = plays, exog= X_list).fit()
print(r.summary())


# backward elemination isleminde zararli olan 4. kolonu kaldiralim ve yeniden egitelim

datas = pd.concat([datas.iloc[:,0:3],datas.iloc[:,4:6]],axis=1)
print(datas)

x_train, x_test, y_train, y_test = train_test_split(datas, plays, test_size = 0.33, random_state= 0 )

# training
r2 = LinearRegression()
r2.fit(x_train,y_train)

y_predict2 = r2.predict(x_test)

print(y_predict2)

X_list = datas.iloc[:,[0,1,2,3,4]].values
r = sm.OLS(endog = plays, exog= X_list).fit()
print(r.summary())

# istenilen test degerlerine biraz daha yaklasti
# OLS raporuna bakilir ve P-Value gore tekrardan x4 kaldirilirsa basari orani azaliyor.
# Her seferinde kolon kaldirmak basari oranini arttirmaz
