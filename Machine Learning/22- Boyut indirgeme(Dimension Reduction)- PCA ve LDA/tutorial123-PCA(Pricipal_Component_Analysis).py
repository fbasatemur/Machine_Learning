# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:51:14 2019

@author: Monster
"""
"""
PCA(Principal Component Analysis):
    Boyut indirgeme ile gereksiz boyutlardan kurtulma veya boyut birlestirme islemlerinde kullanilir
    
Ilk olarak veri kumesinde bir degisiklik yapmadan Customer Segmentasyon un tahmini,
Daha sonra veri kumesinde bagimsiz degiskenleri barindiran 13 kolonu PCA ile indirgeyerek 
Customer Segmentasyon tahmini ve bu tahminler uzerinden PCA in veri uzerindeki tahmin basarisina etkisi degerlendirilecek
"""
import numpy as np
import pandas as pd

datas = pd.read_csv("Wine.csv")

X = datas.iloc[:,0:13].values   # bagimsiz degerler 
Y = datas.iloc[:,13].values     # bagimli degerler

# veri dilimleme
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )

# verilerin normalizasyonu
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# LogisticRegression modeli ile train asamasi
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

y_pred = classifier1.predict(X_test)        # boyut donusumu uygulanmamis tahmin



# PCA boyut donusumu

from sklearn.decomposition import PCA
pca = PCA(n_components = 2 )            # n_components -> azaltilmak istenen kolon sayisi

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)         # X_train de egitilen uzay uzerinden yorumlanmak istendiginden X_test yalnizca transform edilir

# PCA ile donusturulen icin LogisticRegression ile train asamasi
classifier2 = LogisticRegression(random_state= 0)
classifier2.fit(X_train2, y_train)

y_pred2 = classifier2.predict(X_test2)      # boyut donusumu uygulanmis tahmin


# tahminlerin degerlendirilmesi
from sklearn.metrics import confusion_matrix

print("Actula - Pred:")
cm = confusion_matrix(y_test, y_pred)   # Gercek - PCA siz tahmin
print(cm)

print("Actual - PCA Pred:")
cm2 = confusion_matrix(y_test, y_pred2) # Gercek - PCA li tahmin
print(cm2)

print("Pred - PCA Pred:")
cm3 = confusion_matrix(y_pred, y_pred2) # PCA siz - PCA li tahmin
print(cm3)

# PCA uygulanmis ve uygulanmamis verilerin tahmin uzerindeki etkilerine CM uzerinden bakarsak !/36 hata orani oldugu goruldu





