# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:43:27 2019

@author: luning
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


gpcr_length = 1024
gpcr_radius = 4
gpcr_diameter = int(gpcr_radius) * 2



feature = pd.read_csv(r'C:\Users\14420\Desktop\abc\B1_ECFP8_1024.csv',header=None)
response = pd.read_excel(r'C:\Users\14420\Desktop\abc\Response.xlsx',header=None)

X_train, X_test, y_train, y_test = train_test_split(feature, response, random_state=1)

col = []
for i in X_train.columns:
    col.append('D_' + str(i))

ind = []
for i in X_train.index:
    ind.append('M_' + str(i))

X_train.columns = col
X_train.index = ind

y_train.columns = ['Act']
y_train.index = ind
d_train = pd.concat([y_train, X_train], axis=1)
d_train.index.name = 'MOLECULE'
d_train.to_csv(r'C:\Users\14420\Desktop\1024.csv',sep=',')

col = []
for i in X_test.columns:
    col.append('D_' + str(i))

ind = []
for i in X_test.index:
    ind.append('M_' + str(i))

X_test.columns = col
X_test.index = ind

y_test.columns = ['Act']
y_test.index = ind
d_test = pd.concat([y_test, X_test], axis=1)
d_test.index.name = 'MOLECULE'
d_test.to_csv(r'C:\Users\14420\Desktop\10241.csv',sep=',')
