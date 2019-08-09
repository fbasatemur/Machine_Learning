# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:37:43 2019

@author: Monster
"""

"""
k-fold Cross Validation( k katlamali capraz dogrulama ) : 
    Uygulanan yontemin basarisinin olculmesi icin, tum veri kumesinin uzerinde, veriyi egitim ve test kumeleri olarak ayirir.
    Algoritmasi :
        1- ilk olarak k degeri belirle( literaturde tavsiye edilen k=10 secilmesi)
        2- Veri kumesi k degeri kadar parcalara ayrilir
        3- Sirasi ile ilk parca secilir ve secilen parca test, geriye kalan k-1 parca ise train kumesi olarak secilir 
        4- Daha sonra 2. parca test ve geriye kalan k-1 parca train secilir
        5- Bu sekilde k adet test parcasi secilerk her secim sonucunda istenilen algoritma (ornegin bir siniflandirma alg) calistirilir
        ve bir deger elde edilir.
        6- Sistemin basarisi veya error rate(hata orani ) ise elde eilen k adet sonucun ortalamsi alinarak hesaplanir  
"""

import numpy as np
import pandas as pd

datas = pd.read_csv("Social_Network_Ads.csv")

x = datas.iloc[:,[2,3]].values      
y = datas.iloc[:,4].values

# veri dilimleme
from sklearn.model_selection import train_test_split       
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state = 0)
# test_size = 0.25 ise train' e 0.75 kalir; test kumesi tum kumede 4 kez yer degistireceginden k-fold cv degeri 4 olur 

# normalization 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# SVM
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


# k-fold Cross Validation
from sklearn.model_selection import cross_val_score
# estimator -> kullanilacak algoritma (classifier i kullanicaz)
# X -> bagimsiz degerler
# y -> bagimli degerler
# cv -> katlama sayisi(test_size in tum veriyi gezinmesi icin adim sayisi)
score = cross_val_score(estimator= classifier, X= X_train, y= y_train, cv= 4)
print(score.mean())     # ortalama
print(score.std())      # standart sapma 












