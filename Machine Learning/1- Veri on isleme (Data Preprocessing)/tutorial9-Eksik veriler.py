# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:03:02 2019

.csv veri kumsesindeki 'NaN' ile ifade edilen eksik verileri, verinin bulundugu kolonun ortalamasi ile 
doldurarak eksik veri problemini ortadan kaldirabiliriz.
"""

import numpy as np
import pandas as pd

# veri yuklemesi

eksikveriler = pd.read_csv('eksik_veriler.csv')

print(eksikveriler)
# csv dosyasindaki eksik verileri tablo ortalamalari ile doldur

# sci-kit learn ---> sklearn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis=0)
Yas = eksikveriler.iloc[:,1:4].values       # yalnizca sayisal olan kolonlar alinir [satir:satir,kolon:kolon]  
# [:,1:4]-> 1. kolondan 4. kolona kadar (4 dahil degil)
# .values kolon basliklarini almamak icin array donusumudur

print(Yas)
imputer = imputer.fit(Yas[:,1:4])   # 1. 2. ve 3. kolonlar icin ort hesapla
Yas[:,1:4] = imputer.transform(Yas[:,1:4])  # parametre degisikligini uygula
print(Yas)



      
      











