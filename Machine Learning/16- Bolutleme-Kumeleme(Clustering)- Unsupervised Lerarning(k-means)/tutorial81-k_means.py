# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:05:07 2019

@author: Monster
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("musteriler.csv")

X = datas.iloc[:,3:].values   # hacim ve maas degerleri

# k-means
from sklearn.cluster import KMeans
kmeans  = KMeans(n_clusters=3 , init = "k-means++" )    # n_clusters -> k merkez noktalari sayisi
kmeans.fit(X)
# 2 boyutlu olarak 3 cluster icin 3 orta nokta ve kordinatlari
print(kmeans.cluster_centers_)  # ilk kolon hacim sonraki kolon ise maas orta noktalarinin x-y kordinatlari

# WCSS yontemini kullanarak k icin optimum degeri bulalim
# ilk oklarak farkli k degerleri icin loop olusturalim
values = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++", random_state= 123)    # random olustururken ayni noktalar icin olustursun
    kmeans.fit(X)
    values.append(kmeans.inertia_)      # kmeans.inertia_ -> WCSS degerini verir

# WCSS degerlerini gorsellestirelim
plt.plot(range(1,11),values)    # grafige baktigimizda dirsek noktalari olarak 2, 3 veya 4' den biri secilebilir

























