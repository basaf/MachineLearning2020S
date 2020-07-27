# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 08:41:35 2020

@author: guser
"""

import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

from sklearn.model_selection import train_test_split


import numpy as np
from numpy import loadtxt

from tpot import TPOTRegressor
import tpot

dataSetPath='./data/'
#dataSetList=['traffic_volume', 'communities', 'realEstate', 'student_alcohol']
dataSetList=['realEstate']#, 'student_alcohol']


tpot_config = {
    'sklearn.neighbors.KNeighborsRegressor': { 
        'n_neighbors':range(1, 101),
        'weights':['uniform', 'distance'],
        'p': [1, 2]
     },
    'sklearn.ensemble.RandomForestClassifier': { 
    'n_estimators': [100], 
    'criterion': ['gini', 'entropy'], 
    'max_features': np.arange(0.05, 1.01, 0.05), 
    'min_samples_split': range(2, 21),
    'min_samples_leaf': range(1, 21),
    'bootstrap': [True, False] 
    },
    
    'sklearn.neural_network.MLPRegressor': {
         'hidden_layer_sizes': range(1,10),
         'activation':['identity', 'logistic', 'tanh', 'relu'],
         'solver':['lbfgs', 'sgd', 'adam']
        
    },

     'sklearn.linear_model.Ridge': {
     'alpha': np.arange(1e-3, 100.)
    },
     
     'sklearn.svm.SVR':{
        'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'degree':range(2,10),
        'C' :np.arange(1e-2, 100.)
         }
}

#%%
#create path
for fileName in dataSetList:
    y = np.loadtxt(os.path.join(dataSetPath, fileName + '_y'), delimiter=',')
    X = np.loadtxt(os.path.join(dataSetPath, fileName + '_X'), delimiter=',')

    #%% split data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    

    #%%  TPOT
    
    #use template to avoid StackingEstimator!
    
    pipeline_optimizer = TPOTRegressor( template="Regressor" ,population_size=20, cv=5,
                                        random_state=42, verbosity=2,config_dict=tpot_config, max_time_mins=60)#, early_stop=True)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export(fileName + '_tpot_exported_pipeline.py')