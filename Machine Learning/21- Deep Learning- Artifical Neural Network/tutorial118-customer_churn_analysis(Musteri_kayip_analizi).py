# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:27:17 2019

@author: Monster
"""
"""
    Customer churn analysis
    Musteriyi kaybetmeden once, musteriyi kaybedebilecegimizi anlayabilir miyiz ?
    Amac eldeki insani tutabilmek.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
ohe = OneHotEncoder(categorical_features =[1])  # yalnizca 1. kolona uygula
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

# Data preprocessing asamasi bitti, Artifical Neural Network olusturmaya baslayabiliriz.































