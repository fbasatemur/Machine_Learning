# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:35:53 2019

@author: Monster
"""
# 1-kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2- veri on isleme

# 2.1- veri yukleme
datas = pd.read_csv('veriler.csv')

'''
x = datas.iloc[5:,1:4].values
y = datas.iloc[5:,-1:].values

datas -> 0-4 indexli satirlari, cocuk veriler outlier(aykiri,sistemi bozan ) verilerdir.
0-4 indexli satirlari cikarirsak daha basarili tahmin yapacaktir.
'''
x = datas.iloc[:,1:4].values
y = datas.iloc[:,-1:].values
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 0)   
# literaturde test icin 1/3 iken train icin 2/3 kullanilir

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)     # x_train icin ogren ve uygula
X_test = sc.transform(x_test)           # x_test icin sadece uygula

# logistic regresyon -> sayisal degerler uzerinden siniflandirma yapar
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)

print("Actual data:\n",y_test)
y_pred =logr.predict(X_test)
print("Logisctic REG Predict:", y_pred)



# confusion matrix -> siniflandirma degerlerinin tahmini ve actual data'larin kacinin dogru, yanlis siniflandirildigina dair bilgi verir
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)       # parametre sirasi onemsiz
print(cm)
# cm ciktisina gore 1.kolon erkek kolonu,2.kolon kiz kolonu
# C(0,0)-> erkegin dogru tahmini, C(1,1)->kadinin dogru tahmini
# C(1,0)-> erkegin yanlis tahmini, C(0,1)-> kadinin yanlis tahmini



# tutorial: 56

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')   # n_neighbors -> komsu sayisi, metric-> mesafe yaklasimi; euclidean, mahalanobis ...gibi
# minkowski -> sum(|x-y|^p)^(1/p)  
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Predict:",y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# KNN yaklasimi sayesinde daha dogru siniflandirma tahmini yapildi
# KNN de parametre olarak kullanilan n_neighbors ve metric yontem buyuk onem tasimaktadir


















