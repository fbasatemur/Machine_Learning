# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:27:17 2019

@author: Monster
"""
"""
    Customer churn analysis
    Musteriyi kaybetmeden once, musteriyi kaybedebilecegimizi anlayabilir miyiz ?
    Amac eldeki insani tutabilmek.
    Bu olayi Artifical Neural Network kullanarak simule edebiliriz.
"""

import pandas as pd

datas = pd.read_csv("Churn_Modelling.csv")
# verilerin ilk 3 kolonu gereksiz bilgi tasimakta


# Data preprocessing asamasi

X = datas.iloc[:,3:13].values
Y = datas.iloc[:,-1].values

# geography and gender Kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,1] = le1.fit_transform(X[:,1])

le2 = LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features =[1])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

# verileiri dilimleme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.33, random_state = 0)

# verileri normalize et
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Artifical Neural Network 

import keras
from keras.models import Sequential     # ANN olusturmak icin 
from keras.layers import Dense          # ANN katmani (layer ) olustur

classifier = Sequential()               # keras kullanarak bir yapay sinir agi olustur
classifier.add(Dense(6, init = 'uniform' , activation= 'relu', input_dim = 11))    # ANN e layer ekle 
# Dense( units -> Hidden layer daki noron sayisi, 
# init -> initializer ( verileri ilklendir, ANN un islemlendirmesi icin), 
# activation -> kullanilacak aktivasyon fonksiyonu( belirtilmezse linear activation a(x)= x olucaktir),
# relu -> rectifier fuction ( weight 'ler , 0'in altinda 0 ; 0'in ustunde ise linear artan olucak ),
# input_dim -> input layer' daki noron sayisi )

# Tavsiye edilen input layer da linear activasyon fonksiyonu; output layer da ise sigmoid fonk kullanmaktir
classifier.add(Dense(6, init = 'uniform', activation= 'relu'))      # 2. hidden layer
# Eger ki 2. hidden layer eklenirken sorun ile karsilasilirsa
# Tools -> Preferences -> Python interpreter ->  Enable UMR , disable edilerek her seferinde run edildiginde modullerin tekrar yuklenmesi engellenebilir 

classifier.add(Dense(1, init= 'uniform', activation= 'sigmoid'))    # output layer 

# input ve output katmanlaridaki neuron sayilarin ortalamasi ile hidden layer neuron sayilari belirlenebilir  (11+1)/2 = 6
# ancak bu layer lara ait uygun neuron sayilarinin belirlenmesi parametre optimizasyonu bagli olan ayri bir sanat isidir.  
"""
Olusturulan ANN yapisi:
    
input   first   second      output
layer   hidden  hidden      layer
0        0          0        
0        0          0    
.        0          0         0
.        0          0
.        0          0
0        0          0

total amount of neurons
11       6          6         1

"""

# ANN yapisi olusturuldu, simdi ise olusturulan NN'un nasil calisacagini ayarlayalim

# optimizer gibi parametrelerin farkli algoritmalari tensorflow ve ustunde calistigi keras dokumentasyonlarindan bakilablr 
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics=  ['accuracy'])    
# binary_crossentropy -> test edilecek sonuc degerleri binominal(1-0) oldugundan dolayi 

classifier.fit(X_train, y_train, epochs= 50)
# epochs -> tekrar ogrenme miktari
# loss(kayip ) degeri 0 'a yaklastikca modelin test asamasinda basarisi artacaktir
# console da loss degerine bakmak training basarisin arttirilmasinda yardimci olabilir
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)         # predict degerleri yalnizca 1 ya da 0 degerleini almali 
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)       # degerlendirme amacli confusion matrix 
print(cm)













