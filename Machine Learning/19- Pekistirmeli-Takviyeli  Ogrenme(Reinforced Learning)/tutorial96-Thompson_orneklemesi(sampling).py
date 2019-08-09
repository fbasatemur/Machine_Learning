# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 23:36:15 2019

@author: Monster
"""
"""
Thompson Sampling:
    
    Ni1(n) = i. reklam icin o ana kadar 1 gelenlerin toplami
    Ni0(n) = i. reklam icin o ana kadar 0 gelenlerin toplami
    
    Her reklam icin Beta distribution kullanilarak bir rastgele sayi uretilir
    
    Qi(n) = B( Ni1(n)+1, Ni0(n)+1 )
    
    Secim olarak Beta degeri en yuksek olan reklam secilir
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

datas = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000           # ilan gosterimi miktari
ad_N = 10           # ilan sayisi


total = 0           # toplam odul miktari 
selects = []        # secilen ilanlar icin
N1 = [0] * ad_N     # 1 lerin toplami icin 
N0 = [0] * ad_N     # 0 larin toplami icin


for n in range(1,N):    
    ad = 0          # secilen ilan
    max_th = 0      # en yuksek thompson degeri
    for i in range(0,ad_N):
        randBeta = random.betavariate( N1[i] + 1 , N0[i] + 1 )  # Beta dagilimi kullanildi
        if ( randBeta > max_th ):
            max_th = randBeta
            ad = i
            
    selects.append(ad)
    ad_value = datas.values[n,ad]   # n. satir, ad inci sutun degeri
    
    if(ad_value == 1):
        N1[ad] = N1[ad] + 1
    else: 
        N0[ad] = N0[ad] + 1 
    
    total += ad_value
    
print("Toplam odul:")
print(total)

plt.hist(selects)
plt.show()
    
# hist grafigine bakarsak bu veriler icin Thompson Sampling, UCB den daha basarili cikti.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












