# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:04:35 2019

Hedef: iris veri kumesini kullanarak, yaprak siniflarini en yuksek basari ile siniflandirma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

datas = pd.read_excel("Iris.xls")


x = datas.iloc[:,0:4].values
y = datas.iloc[:,-1:].values


# train ve test dilimlemesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0) 

# veri olcekleme
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(x_train)
X_test = std.transform(x_test)



# LogisticRegression gore siniflandirma 
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(random_state = 0)
lgr.fit(X_train, y_train)
y_pred = lgr.predict(X_test)
# print("actual data:",y_test)
# print("predict data:",y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Logistic_Reg:\n",cm) 



# KNN siniflandirma
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric = "minkowski")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)   # confusion matrix ile basari oraninin gozlemlenmesi
print("KNN:\n",cm)



# SVC 
from sklearn.svm import SVC
svc = SVC(kernel = "linear")    # kernel - > linear, poly ,rbf, sigmoid, precomputed
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("SVC:")
print(cm)



# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Naive-Bayes:\n",cm)



# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Decision Tree:")
print(cm)



# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators =10, criterion = "entropy" )
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("RF:\n",cm)

















