# -*- coding: utf-8 -*-

# 1-kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2- veri on isleme

# 2.1- veri yukleme
veriler = pd.read_csv('veriler.csv')
eksikveriler = pd.read_csv('eksik_veriler.csv')


# veri on isleme
boy = veriler[['boy']]
print (boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)


# eksik veriler icin 
from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values="NaN",strategy = "mean",axis=0)

Yasboykilo = eksikveriler.iloc[:,1:4].values       # yalnizca sayisal olan kolonlar alinir [satir:satir,kolon:kolon]
print(Yasboykilo)
imputer = imputer.fit(Yasboykilo[:,1:4])   # her kolon icin ort hesapla
Yasboykilo[:,1:4] = imputer.transform(Yasboykilo[:,1:4])  # parametre degisikligini uygula
print(Yasboykilo)


# encoder : Categorical -> Numeric

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()     
ulke[:,0] = lb.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ulke = veriler.iloc[:,0:1].values
ohe = OneHotEncoder(categorical_features = "all")
ulke= ohe.fit_transform(ulke[:,0:1]).toarray()      # nominal degrelerden kolon bazli degerler olusturuldu
print(ulke)

# print(list(range(22)))    0-21 sayilari verir

# numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])     # OneHotEncoder verilerinden dataframe olusturulur
print (sonuc)

sonuc2 = pd.DataFrame(data = Yasboykilo, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1:].values

print(cinsiyet)
sonuc3 = pd.DataFrame(data = cinsiyet, index=range(22),columns = ['cinsiyet'])
print(sonuc3)


# concat ile dataframe birlestirme
s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)

s2 = pd.concat([s,sonuc3],axis =1)
print(s2)


# amac train ile boyyaskilo yu iceren df (s) ile egitmek ve sonuc3 df'mini bulmasini istiyoruz
# sklearn.cross_validation da kullanilan train_test_split, sklearn.model_selection 'a tasinmis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size = 0.33, random_state = 0)   # s ve sonuc3 df mi parcalanmali
# literaturde test icin 1/3 iken train icin 2/3 kullanilir

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)









