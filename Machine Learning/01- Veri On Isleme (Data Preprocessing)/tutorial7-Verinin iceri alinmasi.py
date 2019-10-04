# -*- coding: utf-8 -*-
"""
Pandas -> dosya okuma islemleri ve veri analiz araçları sağlayan açık kaynaklı bir BSD lisanslı kutuphanedir.
Numpy -> bilimsel hesaplama  işlemleri kolaylaştırmak için yazılmış olan bir python kutuhanesidir.
Matplotlib -> Verileri gorsellestirmede siklikla kullaniln python kutuphanesidir.
"""
import numpy as np                # Suanlik kullanilmayacak
import matplotlib.pyplot as plt   # Suanlik kullanilmayacak
import pandas as pd               # Dosyadan veri cekmek icin kullanilacak

# verilerin yuklenmesi
# pandas library kullanilir

# csv -> comma sepporatted value
datas = pd.read_csv('veriler.csv')    # parametre path ya da name olabiliir
# datas = pd.read_csv("veriler.csv")  python ozelligi geregi "" da kullanilabilir


print(datas)

boy = datas[['boy']]      # boy kolonunu ( dataframe olarak ) al

print(boy)
boykilo = veriler[['boy','kilo']]       # boy ve kilo kolonunu birlikte alabiliriz
print(boykilo)
