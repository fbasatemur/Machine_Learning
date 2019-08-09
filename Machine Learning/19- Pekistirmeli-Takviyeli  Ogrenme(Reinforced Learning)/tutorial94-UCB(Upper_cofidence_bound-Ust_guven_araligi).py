# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:08:24 2019

@author: fbasatemur
"""

'''
UCB( Upper Confidence Bound )
Gecmis bilgilerden ders cikararak bir sonraki tercihi gecmis bilgi dagilimina gore yapan algoritma 

Step 1:
    n adet turun, her i reklami icin
    
    Ni(n) -> i reklaminin o ana kadarki toplam tiklanmasi
    Ri(n) -> o ana kadar i reklamindan gelen toplam odul (Toplam 1 sayisi)
    
Step 2:
    O ana kadarki reklam ortalama odulu => Avarage =  Ri(n) / Ni(n)
    
    Guven araligi oynama potansiyeli    => di(n)   =  ( 3/2 * log(n)/Ni(n) )^ 1/2
    
Step 3:
    UCB = Reklam ort odulu + Guven araligi
    
    En yuksek UCB degerine sahip olan alinir
    
    
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("Ads_CTR_Optimisation.csv" , header = None)


# UCB
import random
import math

N = 10000   # Gosterilme sayisi
ad_N = 10   # reklam sayisi

ad_values = [0] * ad_N   # Ri(n)  Reklam odulleri ilk basta 0 
ad_clicks = [0] * ad_N   # Ni(n)  Reklam tiklamalari ilk basta 0 
avarage = 0     
total = 0       # Toplam odul
selects = []    # tiklananlar
ad = random.randrange(10)        # ilk olarak rastgele bir reklam sec

for n in range(1,N):            # her seferinde gosterilecek reklam secimi (10000 gosterim olucak), kolon basliklarini pas gec (1,N)
    max_ucb = 0
    for i in range(0,ad_N):     # maximum ucb degeri bulunan reklam secilir
        if(ad_clicks[i] > 0):   # tiklanan reklami sec
            avarage = ad_values[i] / ad_clicks[i]       # Reklam ortalama odulu 
            trust_range = math.sqrt(3/2 * (math.log(n) / ad_clicks[i]))       # Guven araligi
            ucb = avarage + trust_range  # ucb degeri
        else:
            ucb = N * 10
            
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    selects.append(ad)                  # secilen reklami secilenler listesine ekle
    ad_clicks[ad] = ad_clicks[ad] + 1   # secilen reklamin tiklamasini arttir
    ad_value = datas.values[n,ad]
    ad_values[ad] = ad_values[ad] + int(ad_value)   # secilen reklamin odul degerini guncelle
    total += int(ad_value)

print("Toplam odul: \n",total)
plt.hist(selects)
plt.show()








































