# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 00:17:05 2019

@author: Monster
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

# sklearn.cross_validation da kullanilan train_test_split, sklearn.model_selection 'a tasinmis
# verilerin train ve test parcalamsi yapilir
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state = 0)   # s ve sonuc3 df mi parcalanmali
# literaturde train icin 2/3 ve test icin 1/3 kullanilir
# arguman olan aylar bagimsiz, satislar ise bagimli degiskenlerdir y = ax+b

'''
# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler    # standartlastirma icin 

sc = StandardScaler()           
X_train = sc.fit_transform(x_train)     # standartlastirma 
X_test = sc.fit_transform(x_test)
Y_train =sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test) 
'''
# CRISP_DM de Modelleme
# model insaa edilir (lineer regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)     # modeli insa ediyor->  x_train ve y_train verdigimizde,ikisi arasindaki birlikteligi anliyor

predict = lr.predict(x_test)  # burada ise x_test verilip, y_test verileri tahmin edilmeye calisiliyor, predict-> tahmin edilen
print(predict)

# veri gorsellestirme
x_train = x_train.sort_index()      # x_train verilerini index e gore siralar 
y_train = y_train.sort_index()      # x_train ve y_train verileri %66.. kismi olarak secildiginden grafikte o kadari gozukur

plt.plot(x_train,y_train)               # x_train ve y_train cizilir ->Blue Line
plt.plot(x_test,lr.predict(x_test))     # x_test ve x_test verildiginde tahin edilen y_test verileri cizlir

# boylelikle verilen en yakin dogrulari cizmis oluyoroz
# Ctrl + I ile istenilen fonk hakkinda yardim alinilabilir

plt.title("aylara gore satislar")
plt.xlabel("Aylar")
plt.ylabel("Satislar")













