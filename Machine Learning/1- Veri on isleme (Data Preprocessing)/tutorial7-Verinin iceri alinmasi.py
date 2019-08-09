# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 00:03:02 2019

@author: Monster
"""
#ders: 6
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# verilerin yuklenmesi
# pandas library kullanilir

#ders:7
# csv -> comma sepporatted value
veriler = pd.read_csv('veriler.csv')    # parametre path ya da name olabiliir
# veriler = pd.read_csv("veriler.csv")  python ozelligi geregi "" da kullanilabilir

print(veriler)

boy = veriler[['boy']]      # boy kolonunu dataframe olarak al

print(boy)
boykilo = veriler[['boy','kilo']]       # boy ve kilo kolonunu birlikte al
print(boykilo)