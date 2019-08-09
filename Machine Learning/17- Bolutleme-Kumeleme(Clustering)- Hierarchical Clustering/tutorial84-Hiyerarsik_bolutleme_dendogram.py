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
plt.show()


# cluster sayisini 4 secerek tekrar egitirsek 
kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 123)
y_predict = kmeans.fit_predict(X)
plt.scatter( X[y_predict == 0,0],X[y_predict == 0,1],s = 50,c = 'red' ) # s-> data point size
plt.scatter(X[y_predict == 1,0],X[y_predict == 1,1],s = 50, c = "blue")
plt.scatter(X[y_predict == 2,0],X[y_predict == 2,1],s = 50, c = "green")
plt.scatter(X[y_predict == 3,0],X[y_predict == 3,1],s = 50, c = "yellow")
plt.title("K-means")
plt.show()




# HC (Hierarchy Clustering)

from sklearn.cluster import AgglomerativeClustering                                 # affinity -> I1,I2, euclidean, manhattan, cosine, precomputed
ac = AgglomerativeClustering(n_clusters= 3,affinity= "euclidean", linkage = "ward") # linkage -> ward(euclidean), complete, average

Y_predict = ac.fit_predict(X)   # cluster sayisi 3 old gore 0,1,2 olarak cluster tahminlerini yapti
print(Y_predict)

plt.scatter( X[Y_predict == 0,0],X[Y_predict == 0,1],s = 50,c = 'red' )
plt.scatter(X[Y_predict == 1,0],X[Y_predict == 1,1],s = 50, c = "blue")
plt.scatter(X[Y_predict == 2,0],X[Y_predict == 2,1],s = 50, c = "green")
plt.title("HC")
plt.show()


# dendrogram visualizing
import scipy.cluster.hierarchy as sch
dentogram = sch.dendrogram(sch.linkage (X, method = 'ward'))
plt.show()


# dendrogram gorseline bakarask k(n_clusters) 2 olarak alinmasi mantikli gozukuyor

ac = AgglomerativeClustering(n_clusters= 2,affinity= "euclidean", linkage = "ward") 

Y_predict = ac.fit_predict(X)   # cluster sayisi 2 old gore 0,1 olarak cluster tahminlerini yapti

plt.scatter( X[Y_predict == 0,0],X[Y_predict == 0,1],s = 50,c = 'red' )
plt.scatter(X[Y_predict == 1,0],X[Y_predict == 1,1],s = 50, c = "blue")
plt.title("HC(cluster = 2)")
plt.show()







