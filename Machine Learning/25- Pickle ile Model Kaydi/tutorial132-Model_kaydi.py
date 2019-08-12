# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:41:21 2019

"""
"""
Modellerin Kaydedilmesi ve Tekrar Kullanilmasi

Machine Learning de bir veri kumesi eğitiminden sonra ( fit() function sonucu ) bir model ortaya cikar. Bu eğitim işlemi dakikalar yada saatlerce sürebilir. 
Eğitim işlemini her program çalıştığında tekrarlamak yerine bir kez eğittikten sonra oluşan modeli bilgisayara kaydedip tekrar tekrar kullanabiliriz. 
Bu islem icin bir cok hazir kutphane vardir:
    1- Pickle
    2- Pmml
    3- Joblib
    ...
Burada KNN Classification Alg kullanilarak olusturulan modelin, 
Pickle ile model kaydi ve model yuklemesi gosterilecektir.
"""

import pandas as pd

url = "http://www.bilkav.com/wp-content/uploads/2018/03/satislar.csv"   # Seklinde adres olarakta veri alimi yapabiliriz

datas = pd.read_csv(url)
datas = datas.values
X = datas[:,0:1]
y = datas[:,1]

# veri dilimleme
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state= 0)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski')   # n_neighbors -> komsu sayisi, metric-> mesafe yaklasimi; euclidean, mahalanobis ...gibi

knn.fit(X_train, y_train)

print(knn.predict(X_test))  # model kaydindan once tahmin edilen veriler 


# Pickle ile model kaydetme

import pickle

model_name = "model.save"

pickle.dump(knn, open(model_name, 'wb'))    # egitim yapilan model, open('model ismi', 'wb' -> writebinary)
# arti,k dosya olusturuldu ve diske kaydedildi

# kayit modelini yukleme

loaded = pickle.load,(open(model_name,'rb'))

print(loaded.predict(X_test))       # model kaydindan sonra tahmin edilen veriler

# boylelikle model kaydini olusturduk ve kayitli bir modeli okuyarak uzerinden tahmin edilenlerin ayni oldugunu gorduk



















