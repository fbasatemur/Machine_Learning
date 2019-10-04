# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:58:33 2019

@author: Monster
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme
datas = pd.read_csv('maaslar.csv')

x = datas.iloc[:,1:2]   # x -> egitim seviyeleri
y = datas.iloc[:,-1:]   # y -> maas miktarlari

X = x.values    # x ve y kolon basliklarindan kurtar yalnizca degerleri al
Y = y.values

# Linear Reg icin training 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)   # egitim

# cizme islemleri
plt.scatter(X,Y, color = 'blue')    # egitim ve maaslari point olarak ciz
plt.plot(x,lin_reg.predict(X), color='green')  # egitim degerleri verildiginde egtmden tahmin edilen maaslari linear olark ciz
plt.show()          # tekrardan farkli bir grafik uzerinden gosterim icin

# simdi ise polylominal regression ile egitim yapalim

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)   # 2. der polinom object create

X_poly = poly_reg.fit_transform(X)  # train edilecek veri ogretilmeden once polinomal veriye donusturulur 
print(X_poly)
# polinomal veri linear reg verilerek egitilir
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)     # ogrenme
plt.scatter(X,Y,color = 'green')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.show()


# tahminler
# linear reg tahminleri 
print(lin_reg.predict(1))
'''
print(lin_reg.predict(6.6))

# Polynomial reg tahminleri
print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))
'''