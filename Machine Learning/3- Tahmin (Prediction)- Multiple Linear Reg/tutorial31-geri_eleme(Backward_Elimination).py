# -*- coding: utf-8 -*-

# 1-kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2- veri on isleme

# 2.1- veri yukleme
veriler = pd.read_csv('veriler.csv')

# encoder : Kategoric -> Numeric

ulke = veriler.iloc[:,0:1].values
print(ulke)

yasboykilo = veriler.iloc[:,1:4].values
print(yasboykilo)

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

# ulke ve cinsiyet onehotencoder ile nominal degerlerden numeric donusumu yapilir

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = "all")
ulke= ohe.fit_transform(ulke).toarray()      # nominal degrelerden kolon bazli degerler olusturuldu
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = "all")
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

# print(list(range(22)))    0-21 sayilari verir

# numpy dizileri dataframe donusumu

sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])     # OneHotEncoder verilerinden dataframe olusturulur
print (sonuc)

sonuc2 = pd.DataFrame(data = yasboykilo, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet[:,:1], index=range(22),columns = ['cinsiyet'])    # Dummy Variable i engellemek icin yalnizca 1 kolon alinir
print(sonuc3)

# concat ile dataframe birlestirme

s = pd.concat([sonuc,sonuc2],axis = 1)

s2 = pd.concat([s,sonuc3],axis = 1)
print(s2)

# amac train ile boyyaskilo yu iceren df (s) ile egitmek ve sonuc3 df'mini bulmasini istiyoruz
# sklearn.cross_validation da kullanilan train_test_split, sklearn.model_selection 'a tasinmis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)   # s ve sonuc3 df mi parcalanmali
# literaturde test icin 1/3 iken train icin 2/3 kullanilir

# verilerin olceklenmesi
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
'''
# multiple variable icin model egitimi cinsiyet kolonu icin yapiliyor

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)  # egitim yapiliyor

y_pred = regressor.predict(x_test)  # x_test veilerek, y_test tahmin edilmeye calisiliyor
# farkli bir kolon uzeinden multiple linear reg modeli olusturup backware elimination ile verimli bir regreyon modeli elde edelim


# boy kolonu uzerinden egitilmesini istiyoruz diyelim..
# boy kolonunu at

boy = s2.iloc[:,3:4].values
print(boy)

boy_sol = s2.iloc[:,:3]
boy_sag = s2.iloc[:,4:]

veri = pd.concat([boy_sol,boy_sag],axis=1)  # boy kolonu atildi ve birlestirildi

# simdi egitim verilerini hazirlayalim ve parcalayalim..

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state = 0)    # veri bagimsiz, boy ise tahminl edilmesi istenilen bagimli degisken 

r2 = LinearRegression()
r2.fit(x_train,y_train)  # egitim yapiliyor

y_pred = r2.predict(x_test)  # x_test veilerek, y_test tahmin edilmeye calisiliyor

#  ders: 31 

''' 
 y = b0+ b1x1+ b2x2+ b3x3+ E
 formulundeki b0 degerlerini veri dataframe i ile kolon olarak birlestirip bir liste olusturalim
'''
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1 )    # np.ones() 22 satir,1 kolondan ve 1' lerden olusan, astype() par. turunde dizi verir
# bu X dizisi veri' ye eklenir ve axis = 1 -> kolon olarak eklenir, 0-> satir

# simdi veri df deki her bir kolonu ifade eden bir liste olusturalim
X_list = veri.iloc[:,[0,1,2,3,4,5]].values
r = sm.OLS(endog = boy, exog = X_list).fit() # endog -> bagimli, exdog -> bagimsiz degsken
# X_list deki her bir kolonun boy kolonu uzerindeki p-value degerleri hesplanir
print(r.summary())

# backward eliminatin a gore en yuksek P deg sahip olan deger elenir yani burada en yuksek P degsahip kolon elenicek x5 kolonu yani 4 indisli kolon elenir

X_list = veri.iloc[:,[0,1,2,3,5]].values
r = sm.OLS(endog = boy, exog = X_list).fit()
print(r.summary())
# 5 indisli elenir

X_list = veri.iloc[:,[0,1,2,3]].values
r = sm.OLS(endog = boy, exog = X_list).fit()
print(r.summary())

# En sonunda tum P degerleri 0 olana kadar hangi kolonun ise yarayip yaramadigina dair bir regresyon modeli cikarilmis olur
















