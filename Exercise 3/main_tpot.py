# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 08:41:35 2020

@author: guser
"""

import pandas as pd
import matplotlib.pyplot as plt
#import configuration as cfg
import os

from sklearn.model_selection import train_test_split


import numpy as np
from numpy import loadtxt

from tpot import TPOTRegressor
import tpot

dataSetPath='./DataFiles/'
dataSetList=['traffic_volume']

#%%
#create path
for fileName in dataSetList:
    path=dataSetPath+fileName
    y=loadtxt(path + '_y', delimiter=',')
    X=loadtxt(path + '_X', delimiter=',')
   
    #%% split data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    

    #%%    
    
    pipeline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2,config_dict=tpot.config.classifier_config_dict_light)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export(fileName + '_tpot_exported_pipeline.py')