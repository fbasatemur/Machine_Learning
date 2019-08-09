# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 00:28:19 2019

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
x = datas.iloc[:,1:4].values    # bagimsiz degiskenler
y = datas.iloc[:,-1:].values    # bagimli degisken

# egitim ve test icin verilerin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 0)   
# literaturde test icin 1/3 iken train icin 2/3 kullanilir

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)     # x_train icin ogren ve uygula
X_test = sc.transform(x_test)           # x_test icin sadece uygula

# Siniflandirma algoritmalari ... 

# 1- logistic regresyon -> sayisal degerler uzerinden siniflandirma yapar
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train, y_train)      # train

print("Actual data:\n",y_test)
y_pred =logr.predict(X_test)    # predict
print("Predict:",y_pred)



# Confusion matrix -> Predict verilerinin, actual data ile uygunlugunu kontrol ederek tahmin verilerinin basarisi gozlemlenebilir
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)       
print("CM:\n",cm)
# cm ciktisina gore 1.kolon erkek kolonu,2.kolon kiz kolonu
# C(0,0)-> erkegin dogru tahmini, C(1,1)->kadinin dogru tahmini
# C(1,0)-> erkegin yanlis tahmini, C(0,1)-> kadinin yanlis tahmini



# tutorial: 56

# 2- KNN (K Nearest Neighborhood)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')   # n_neighbors -> komsu sayisi, metric-> mesafe yaklasimi; euclidean, mahalanobis ...gibi
# minkowski metrik yaklasimi -> sum(|x-y|^p)^(1/p)  
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Predict :\n",y_pred)
cm = confusion_matrix(y_test, y_pred)   # confusion matrix ile basari oraninin gozlemlenmesi
print(cm)
# KNN yaklasimi sayesinde daha dogru siniflandirma tahmini yapildi
# KNN de parametre olarak kullanilan n_neighbors ve metric yontem siniflandirmada buyuk onem tasimaktadir



# tutorial: 58

# 3- SVC (SVM Classifier)
from sklearn.svm import SVC
# diger kernel parametreleri kullanilarak basari orani confusion matrix kullanilarak olculebilir ve arttirilabilr
svc = SVC(kernel = "rbf")    # kernel - > linear, poly, rbf, sigmoid, precomputed
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVC:')
print(cm)
# bu veriler icin, her kernel parametresi denendiginde en iyi basari rbf argumaninda elde edilmistir



# tutorial: 62

# 4- Naive Bayes

# Gaussian Naive Bayes,Multinominal NB, Bernoulli NB cesitlerinden herhangi biri denenebilir
# Gaussian NB -> Tahmin edilecek veri kolon ya da sinif surekli (reel) sayilar ise kullanilabilir.
# Multinominal NB -> Tahmin edilecek veri nominal degerler ise (araba markalari, ulke isimleri gibi..) 
# bu degerler numarandirilarak kullanilabilir
# Bernoulli NB -> binominal yani 1 ya da 0 gibi 2 drum soz konusu ise kullanilabilir 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# Gaussian NB kullanidi ve egitildi..
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)   # basari orani olcumu icin Confusion matrix
print("GNB:")
print(cm)



# tutorial: 64

# 5- Decision Tree 

# daha hassas ogrenim icin "gini" yerine "entropy" kullanildi..
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy")

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("DTC:")
print(cm)




# tutorial: 66

# 6- Random Forest 
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("RFC:")
print(cm)



# tahmin olasililklarin degerlendirilmesi icin 
y_predict_p = rfc.predict_proba(X_test)    # tahmin olasiliklari
print(y_test)
print(y_predict_p[:,0])



# ROC, FPR, TPR, degerleri
from sklearn import metrics 
fpr, tpr, threshold = metrics.roc_curve(y_test, y_predict_p[:,0], pos_label = 'e')
print(fpr)
print(tpr)











