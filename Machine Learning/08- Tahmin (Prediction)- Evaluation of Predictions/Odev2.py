# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:03:47 2019

@author: Monster
"""
# kutuphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm        # p-value  ve R_squared OLS
from sklearn.metrics import r2_score 

datas = pd.read_csv("maaslar_yeni.csv")

# P-value degerleri kontrol edilerek en gerekli kolonlar 2:3 seklinde secildi
unvansKidemPuan = datas.iloc[:,2:5]
maas = datas.iloc[:,-1:]
X = unvansKidemPuan.values
Y = maas.values

# Correlation matrisine gore parametre elemesi yapilabilir
print(datas.corr())

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# p-value degerleri incelenecek
print("Linear reg OLS:")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
print("Linear R-square value:")
print(r2_score(Y , lin_reg.predict(X)))



# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# p-value degerleri incelenecek
print("Polynomial Reg OLS:")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())
print("Polynomial R-square value:")
print(r2_score(Y , lin_reg2.predict(poly_reg.fit_transform(X))))



# SVR 
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli, y_olcekli)
print("SVR OLS:")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
print("SVR R-square value:")
print(r2_score(Y , svr_reg.predict(x_olcekli)))



# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_r = DecisionTreeRegressor(random_state=0)
dt_r.fit(X,Y)
print("Decision Tree OLS:")
model4 = sm.OLS(dt_r.predict(X),X)
print(model4.fit().summary())
print("Decision Tree R-square value:")
print(r2_score(Y , dt_r.predict(X)))


# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_r = RandomForestRegressor(n_estimators=10 ,random_state = 0)
rf_r.fit(X,Y)
print("Random Forest Reg OLS:")
model5 = sm.OLS(rf_r.predict(X),X)
print(model5.fit().summary())
print("Random Forest R-square value:")
print(r2_score(Y , rf_r.predict(X)))





















