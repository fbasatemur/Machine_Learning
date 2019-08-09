# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# verilerin yuklenmesi
# pandas library kullanilir

# csv -> comma sepporatted value
datas = pd.read_csv('veriler.csv')    # parametre path ya da name olabiliir
# veriler = pd.read_csv("veriler.csv")  python ozelligi geregi "" da kullanilabilir

print(datas)

boy = datas[['boy']]      # boy kolonunu ( dataframe olarak ) al

print(boy)
boykilo = veriler[['boy','kilo']]       # boy ve kilo kolonunu birlikte alabiliriz
print(boykilo)
