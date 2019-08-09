# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:35:48 2019

@author: Monster
"""
"""
XGBoost 
Tek bir model yerine birden fazla zayif model kullanarak Loss function larini optimize eden 
ve bunun sonucu olarak loss degerlerini dusuren gradient boosting mekanizmasidir.

import icin:
pip install xgboost  

ya da   https://xgboost.readthedocs.io/en/latest/build.html 
adresinden Installation Guide asamalari takip edilebilir
"""

import numpy as np
import pandas as pd

datas = pd.read_csv("Churn_Modelling.csv")

X = datas.iloc[:,3:13].values
Y = datas.iloc[:,13].values 

# kategorik -> Numeric
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:,1]= lb.fit_transform(X[:,1])
lb2 = LabelEncoder()
X[:,2] =lb2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])   
X = ohe.fit_transform(X).toarray()
X = X[:,1:]



print(X)
# dilimleme
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# normalization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, y_pred)
print(cm)

# confusion matrix degerlerine bakilarak diger Classification algoritmalari ile basarisi kiyaslanabilir.




