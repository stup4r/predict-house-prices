#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:59:47 2018

@author: kinase
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def separateVars(df):
    
    numcolmns=[]
    catcolmns=[]
    for colmn in df.columns:
        if np.issubdtype(df[colmn].dtype, np.number):
            numcolmns.append(colmn)
        else:
            catcolmns.append(colmn)
            
    return numcolmns, catcolmns

def categEncoder(df, target):
    _, catcolmns = separateVars(df)
    df.is_copy=False
    finalDict = {}
    for i in catcolmns:
        df2 = df.groupby(i)[target].mean()
        df2 = df2.sort_values()
        df2 = pd.DataFrame(df2)
        df2['order'] = range(1, 1+df2.index.shape[0])
        df2 = df2.drop(target, axis=1)
        df2 = df2['order'].to_dict()
        df[i].replace(df2, inplace=True)
        finalDict[i] = df2
    return finalDict

def rmse_cv(model, X_data, y_data):
    rmse= np.sqrt(-cross_val_score(model, X_data, y_data, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

def fillinMostFrequent(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print('Filling out the following columns: ')
    print(missing)
    for i in missing.index:
        df[i].fillna(df[i].value_counts().idxmax(), inplace=True)
        
    print('Done...')
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print('The following columns have missing values: ')
    print(missing)