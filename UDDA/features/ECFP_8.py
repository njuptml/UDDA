# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 19:37:34 2017

@author: Jake
"""
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
import xlrd

def smiles_to_fps(data, fp_length, fp_radius):
    return stringlist2intarray(np.array([smile_to_fp(s, fp_length, fp_radius) for s in data]))

def smile_to_fp(s, fp_length, fp_radius):
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(
            m, fp_radius, nBits=fp_length, invariants=[1]*m.GetNumAtoms(), useFeatures=False)).ToBitString()

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array.'''
    return np.array([list(s) for s in A], dtype=int)


#excel = xlrd.open_workbook('D:\shiyan\demo\ligandI.xls')
excel = xlrd.open_workbook(r'C:\Users\14420\Desktop\abc\Input_Smiles.xlsx')
#获取第一个sheet
sheet = excel.sheets()[0]

#打印第j列数据
x1 = sheet.col_values(0)

a1 = smiles_to_fps(x1, 1024, 4)

import csv

#f = open('B1_ECFP8_1024.csv',mode='wb+')
with open('B1_ECFP8_1024.csv','w',encoding='utf8') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(a1)
    f.close()

