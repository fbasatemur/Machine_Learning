# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 23:15:41 2019

@author: Monster
"""
# 10 farkli reklamin tiklanma verileri uzerinde Random selection uygulandiginda hangi reklamin daha basarili oldugu olculecek.
# rastgele secilen reklamlarin tiklanma degerleri toplanacak

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv("Ads_CTR_Optimisation.csv")

# random sayi uretici
import random
N = 10000           # Gosterilme sayisi -> datas satir miktari 
ad_N = 10           # reklam sayisi -> datas kolon syisi
total_click = 0
secilenler = []
for n in range(0,N):
    ad = random.randrange(ad_N)     # rastgele bir reklam secildi
    secilenler.append(ad) 
    click = datas.values[n,ad]      # secilen satir ve kolondaki deger
    total_click += click

# hic bir sekilde ogrenme ve karar verme belirtisi icermez !
# yalnizca random reklam verir 
# en fazla hangisinin rastgele secilgini gorelim
plt.hist(secilenler)
plt.show()





