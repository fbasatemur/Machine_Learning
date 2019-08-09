# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:57:00 2019

@author: fatih
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# veri yukleme

veriler = pd.read_csv('veriler.csv')

# sci-kit learn ---> sklearn
from sklearn.preprocessing import Imputer
# veriler ' den nominal veri kolonu olan ulke kolonu numeric degerlendirme icin dilimleniyor
ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder      # her bir ifadeye sayisal deger atar

le = LabelEncoder()         # le nesnesi tanimlandi
ulke[:,0] = le.fit_transform(ulke[:,0]) 
print(ulke)
                        
from sklearn.preprocessing import OneHotEncoder     # kolon bazli olarak kolon basliginda 1-0 koyarak ilerliyor
ohe = OneHotEncoder(categorical_features='all')                 
ulke = ohe.fit_transform(ulke[:,0:1]).toarray()
print(ulke)

      
      











