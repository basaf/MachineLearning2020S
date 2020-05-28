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
rawData = pd.read_csv("DataSets/Mushroom/mushrooms.csv")

#data analysis
#check for missing values

#show number of different output values


#check distribution

