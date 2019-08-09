# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:03:02 2019

@author: fatih
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme

eksikveriler = pd.read_csv('eksik_veriler.csv')

print(eksikveriler)
# csv dosyasindaki eksik verileri tablo ortalamalari ile doldur

    # sci-kit learn --->
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy = "mean",axis=0)
Yas = eksikveriler.iloc[:,1:4].values       # yalnizca sayisal olan kolonlar alinir [satir:satir,kolon:kolon]
print(Yas)
imputer = imputer.fit(Yas[:,1:4])   # her kolon icin ort hesapla
Yas[:,1:4] = imputer.transform(Yas[:,1:4])  # parametre degisikligini uygula
print(Yas)



      
      











