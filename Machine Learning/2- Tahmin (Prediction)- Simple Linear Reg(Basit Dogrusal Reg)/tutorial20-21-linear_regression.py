# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:02:15 2019


"""

# 1-kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2- veri on isleme

# 2.1- veri yukleme
veriler = pd.read_csv('satislar.csv')

# veri on isleme
aylar = veriler[['Aylar']]
print (aylar)

satislar = veriler[['Satislar']]
print(satislar)


# amac train ile boyyaskilo yu iceren df (s) ile egitmek ve sonuc3 df'mini bulmasini istiyoruz
# sklearn.cross_validation da kullanilan train_test_split, sklearn.model_selection 'a tasinmis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0)   # s ve sonuc3 df mi parcalanmali
# literaturde test icin 1/3 iken train icin 2/3 kullanilir

# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler    
# standartlastirma 
sc = StandardScaler()           
X_train = sc.fit_transform(x_train)     
X_test = sc.fit_transform(x_test)
Y_train =sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test) 

# CRISP_DM de Modelleme
# model insaasi (lineer regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)     # modeli insa ediyor->  x_train ve y_train verdigimizde,ikisi arasindaki birlikteligi anliyor

predict = lr.predict(x_test)  # burada ise x_test verilip, y_test verileri tahmin edilmeye calisiliyor, predict-> tahmin edilen
print(predict)
















