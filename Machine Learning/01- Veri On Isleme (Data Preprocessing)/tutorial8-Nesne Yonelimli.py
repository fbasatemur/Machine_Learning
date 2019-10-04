# -*- coding: utf-8 -*-

# Burada ornek olarak class ve object uretimi gosterilmistir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class person:               # person sinifi ve icinde tanimli run fonksiyonu
    boy = 180
    def run(self,b):        # self parametresi disaridan erisim icin gerkli
        return b+10

ali = person()      # object decleration
print(ali.boy)      # variable access
print(ali.run(30))
list = [1,2,3,4]    # list decleration












