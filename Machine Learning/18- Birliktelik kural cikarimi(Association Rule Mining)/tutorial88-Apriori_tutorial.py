# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:15:12 2019

@author: Monster
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("sepet.csv",header = None)  # ...csv dosyasinin header i yok 

t =[]       # transactions list
'''
Apriori Arguments:
transactions -- A transaction iterable object
(eg. [['A', 'B'], ['B', 'C']]).

list of list mantigi ile her bir satir icin listelerden olusan transaction listesi olusturulmali 
'''
for i in range(0,7501):     # satir sayisi 7501
     t.append( [str(datas.values[i,j]) for j in range (0,20)] )     # bir satirdaki max urun sayisi
    

# apriori.py -> apriori algorthm  
from apyori import apriori
kurallar = apriori(t, min_support =0.01, min_confidence =0.2, min_lift =3, min_lenght = 2)     # min_lenght -> en az birliktelik mikrtari exp: 2 li urun satisi yapmak gibi
'''

Support(b) = b varligini iceren olaylar / Toplam olaylar
Confidence(a->b) = a dan sonra b olayi gerceklesmeleri / Toplam a olaylari     ---> a olayindan sonra b olayi gerceklesiyorsa 
Lift(a->b) = Confidence(a->b) / Support(b)            ---> Lift, iki olayin birbirine olan baglilik degeridir

t-> transaction list
min_support -> veri kumesi fazla oldugundan urun cesitililigi fazladir, bundan dolayi dusuk secildi 
min_confidence -> 0.2 alti alinmaz
min_lift -> minimum baglilik duzeyinin arttirilmasi daha sık ard arda alinan urunlleri verir 
min_lenght -> en az bagli olmasi gereken urun sayisi exp: icki ve bebek bezi gibi 2 urun
'''
print(list(kurallar))

# Eclat alg ise kucuk verilerde daha hizlidir .Ancak veri miktari artarsa Apriori daha uygundur.















