# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:01:10 2020

@author: guser
"""

from arff import load
import configuration as cfg
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap

from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import helper
import functions


#%%
rawData = pd.read_csv( os.path.join(cfg.default.mushroom_data, 'mushrooms.csv'))
#%%
#data analysis
#check for missing values
rawData.isnull().values.any()
rawData.isnull().sum()

#show number of different values for each column

for col in list(rawData):
    print(col)
    l=rawData[col].unique()
    print(rawData[col].value_counts(normalize=True))
    #print(l)
    #print(len(l))

figure, ax = plt.subplots(len(rawData.columns), 1, tight_layout=True, figsize=(7,15))

for num, column in enumerate(rawData.columns):
    p = ax[num]
    p.bar(rawData[column].value_counts(normalize=True).index, rawData[column].value_counts(normalize=True).values )
    
    p.set_title(column)
    p.grid(False)



#%% remove veil-type -> only one value
 data=rawData.drop(labels="veil-type",axis=1)   

# 30% have no value for stalk-root -> only ?

#%% encode data edible=e /1 , poisonous=p /0
 
 y=data['class'].replace({'p':0,'e':1})
 
 X=data.drop(labels="class",axis=1)
 X=X.astype('category')
 
 X_encoded=X
 for colName in X.columns:
     X_encoded[colName]= X[colName].cat.codes
     
#%%
test_size = 0.2
random_state = 1
#%% k-nn
list_k = [1, 3]  # 5, 10, 20, 50, 100, 300]
functions.knn(X_encoded, y, test_size, random_state, list_k, True,
    ['uniform', 'distance'],
    ['holdout',
    'cross-validation'],
    cfg.default.mushroom_figures,
    'knn')