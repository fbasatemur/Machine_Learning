# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:58:33 2019

@author: Monster
"""
# kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme
datas = pd.read_csv('maaslar.csv')

# data frame dilimleme
x = datas.iloc[:,1:2]   # x -> egitim seviyeleri
y = datas.iloc[:,-1:]   # y -> maas miktarlari

# NumPY array donusumu
X = x.values    # x ve y kolon basliklarindan kurtar yalnizca degerleri al
Y = y.values

# test ve train icin veriler test_train_split ile parcalamadan direk tumunu kullanarak egitim ve test yapiliyor
# Linear Reg icin training 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)   # egitim


# simdi ise polylominal regression ile egitim yapalim
# dogrusal olmayan(nonlinear model) olusturma
# 2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)   # 2. der polinom object create
X_poly = poly_reg.fit_transform(X)  # train edilecek veri ogretilmeden once polinomal veriye donusturulur 
print(X_poly)
# polinomal veri linear reg verilerek egitilir
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)     # training


# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
X_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y)


# linear reg verilerin tahmini
print(lin_reg.predict(X))
# 2. dereceden polinomial reg verilerin tahmini 
print(lin_reg2.predict(poly_reg.fit_transform(X)))
# 4. dereceden polinomial reg verilerin tahmini
print(lin_reg3.predict(poly_reg3.fit_transform(X)))


# Gorsellestirme
plt.scatter(X,Y, color = 'red')    # egitim ve maaslari point olarak ciz
plt.plot(x, lin_reg.predict(X), color='blue')  # egitim degerleri verildiginde egtmden tahmin edilen maaslari linear olark ciz
plt.show()          # tekrardan farkli bir grafik uzerinden gosterim icin

plt.scatter(X,Y, color = 'blue')    # egitim ve maaslari point olarak ciz
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='green')  # egitim degerleri verildiginde egtimden tahmin edilen maaslari polinomial olark ciz
plt.show()  
# poly_reg.fit_transform(X) -> yerine direk olarak yukarida zaten ayni sekilde hesapladigimix X_poly verilebilir

plt.scatter(X,Y, color = 'blue')    # egitim ve maaslari point olarak ciz
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color='green')  # egitim degerleri verildiginde egtimden tahmin edilen maaslari polinomial olark ciz
plt.show()  


# tahminler
# linear reg tahminleri 

'''
print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

# Polynomial reg tahminleri
print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))
'''
