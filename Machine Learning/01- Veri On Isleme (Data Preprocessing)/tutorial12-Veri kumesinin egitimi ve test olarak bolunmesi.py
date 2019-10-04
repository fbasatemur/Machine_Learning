# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 23:30:13 2019

Belirli onislemler ile duzenlenen veri kumesini, train ve test icin train_test_split ile bolumlemeliyiz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')
eksikveriler = pd.read_csv('eksik_veriler.csv')

from sklearn.preprocessing import Imputer # eksik veriler icin 

imputer = Imputer(missing_values="NaN",strategy = "mean",axis=0)

Yasboykilo = eksikveriler.iloc[:,1:4].values       # yalnizca sayisal olan kolonlar alinir [satir:satir,kolon:kolon]
print(Yasboykilo)
imputer = imputer.fit(Yasboykilo[:,1:4])   # her kolon icin ort hesapla
Yasboykilo[:,1:4] = imputer.transform(Yasboykilo[:,1:4])  # parametre degisikligini uygula
print(Yasboykilo)

# categorical -> numeric
from sklearn.preprocessing import OneHotEncoder

ulke = veriler.iloc[:,0:1].values
ohe = OneHotEncoder(categorical_features = "all")
ulke= ohe.fit_transform(ulke[:,0:1]).toarray()      # nominal degrelerden kolon bazli degerler olusturuldu

print(ulke)


sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])     # OneHotEncoder verilerinden dataframe olusturulur
print (sonuc)

sonuc2 = pd.DataFrame(data = Yasboykilo, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1:].values

print(cinsiyet)
sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22),columns = ['cinsiyet'])
print(sonuc3)

# concat ile kolonlari birlestir

s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)

s2 = pd.concat([s,sonuc3],axis =1)
print(s2)

# amac train ile boyyaskilo yu iceren dataframe(s) egitmek ve sonuc3 df'mini bulmasini istiyoruz

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)   # s ve sonuc3 df mi parcalanmali
# literaturde test icin 1/3 iken train icin 2/3 kullanilir
















