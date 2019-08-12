# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:57:00 2019

Veri kumemizde kategorik veriler de bulunabilir. Bu verileri numeric verilere donusturmemiz gerekir.
Bunu icin sklearn den LabelEncoder ve OneHotEncoder i kullanbiliriz 
LabelEncoder -> 0 dan baslayarak ayni kolonda bulunan verileri categorical -> numeric donusumunu yapar.
OneHotEncoder -> Ayni kolonda bulunan categorical verilerin her birini bir kolon olarak ekler ve 1-0 olarak doldurur.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme

datas = pd.read_csv('veriler.csv')

# sci-kit learn ---> sklearn
from sklearn.preprocessing import Imputer

# alinan veriler, nominal veri kolonu olan ulke kolonunu numeric degerlendirme icin dilimleniyor
ulke = datas.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder      # her bir ifadeye sayisal deger atar

le = LabelEncoder()                                 # le nesnesi tanimlandi
ulke[:,0] = le.fit_transform(ulke[:,0]) 
print(ulke)
                        
from sklearn.preprocessing import OneHotEncoder     # kolon bazli olarak kolon basliginda 1-0 koyarak ilerliyor
ohe = OneHotEncoder(categorical_features='all')                 
ulke = ohe.fit_transform(ulke[:,0:1]).toarray()
print(ulke)

      
      











