# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:37:43 2019

@author: Monster
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
print("Score: ",score.mean())     # ortalama
print("Std: ",score.std())      # standart sapma 


# Grid search ile parametre optimizasyonu
# SVC ile tanimlanan classifier nesnesinin parametrelerini optimize etmeye calisicaz
from sklearn.model_selection import GridSearchCV
# Grid search e verilmek istenilen parametreler bir liste olarak yazilir. 
parameters = [ {'kernel':['linear'], 'C':[1,2,3,4,5] }, 
      { 'C':[1,10,100], 'kernel':['rbf'],  
       'gamma': [1, 0.5, 0.1, 0.01, 0.001] } ]

"""
estimator -> siniflandirma algoritmasi (optimize edilecek algoritma nesnesi)
param_grid -> parameters
scoring -> skorlama kriteri (accuracy, ...)
cv -> katlama sayisi
n_jobs -> ayni anda calisacak is
"""
gs = GridSearchCV(estimator= classifier, 
                  param_grid = parameters, 
                  scoring= 'accuracy', 
                  cv= 10, 
                  n_jobs= -1)

grid_search = gs.fit(X_train, y_train)

best_score = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best score: ", best_score)
print("Best params: ", best_parameters)

# Parametre optimizasyonu yapmadan elde edilen 'Score ' degeri, Grid search optimizasyonu sayesinde  0.3 artti  
# En iyi parametrelere bakarsak linear kullanilmayabilir; gamma ve rbf kullanilabilir
















