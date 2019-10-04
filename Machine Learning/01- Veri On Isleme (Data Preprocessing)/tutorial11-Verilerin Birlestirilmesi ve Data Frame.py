# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')
eksikveriler = pd.read_csv('eksik_veriler.csv')

from sklearn.preprocessing import Imputer # eksik veriler icin 

# eksikleri doldurma yontemi -> 'mean'
imputer = Imputer(missing_values="NaN",strategy = "mean",axis=0)

Yasboykilo = eksikveriler.iloc[:,1:4].values              # 1. 2. ve 3. kolonlari dilimle
print(Yasboykilo)
imputer = imputer.fit(Yasboykilo[:,1:4])                  # her kolon icin ort hesapla
Yasboykilo[:,1:4] = imputer.transform(Yasboykilo[:,1:4])  # 1. 2. ve 3. satirdaki eksik verileri kolon ortalamalri ile doldur
print(Yasboykilo)

# categorical -> numeric transform
from sklearn.preprocessing import OneHotEncoder

ulke = veriler.iloc[:,0:1].values
ohe = OneHotEncoder(categorical_features = "all")
ulke= ohe.fit_transform(ulke[:,0:1]).toarray()            # nominal degrelerden kolon bazli degerler olusturuldu

print(ulke)

# print(list(range(22)))    0-21 sayilarindan olusan liste verir
# columns -> kolon basliklari
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])     # OneHotEncoder verilerinden dataframe olusturulur
print (sonuc)

sonuc2 = pd.DataFrame(data = Yasboykilo, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1:].values       # -1. yani son kolon (bagimli degerleri icerir)

print(cinsiyet)
sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22),columns = ['cinsiyet'])
print(sonuc3)

# concat ile kolonlari birlestir

s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)

s2 = pd.concat([s,sonuc3],axis =1)
print(s2)




















