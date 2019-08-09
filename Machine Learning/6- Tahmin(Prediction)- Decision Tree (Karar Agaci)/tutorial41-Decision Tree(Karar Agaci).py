# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 23:10:38 2019

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
X = x.values
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
# ders 39

# verilerin olceklenmesi
# SVR yapmadan once veriler StandardScaler dan gecirilmeli
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

# SVR -> Support Vector Regression -> min marjinal araligi ile maximum data frame'mi araraliga almayi hedefleyen vector
# buna bagli ortayan atilan fonksiyona SVR denir
from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')       # radial basics functions -> Gauss Teoremi
# rbf den farkli olarak bir cok kernel function bulunur
svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color = 'red')
plt.plot(x_olcekli, svr_reg.predict(x_olcekli),color = 'blue')
plt.show()      # bu plotu ciz daha sonraki plotu ust uste cizmesini engeller
'''
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))
'''
# Decision Tree icin veri olceklemeye gerek yok
# Decsision tree yalnizca tablodan verilen degerleri dondurur, herhenagi bir ara deger tahmini yapmaz
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z = X+ 0.5
K = X- 0.4
plt.scatter(X,Y, color='green')
plt.plot(X, r_dt.predict(X),color = 'blue')
plt.plot(X, r_dt.predict(Z), color='red')
plt.plot(X, r_dt.predict(K),color = 'yellow')
plt.show()

print(r_dt.predict(11))
print(r_dt.predict(6.6))














