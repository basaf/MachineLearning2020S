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
 
X=X_encoded.to_numpy()
y=y.to_numpy()    
#%%
test_size = 0.2
random_state = 1

#%% Data scaling (remove mean and scale to unit variance)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)


validation_methods = ['holdout', 'cross-validation']
baselines=['stratified', 'uniform']

#%% k-Nearest Neighbor Classification
if True:
    list_k = [1,3,5, 10, 50]
    weights = ['uniform', 'distance']

    functions.knn(X, y, test_size, random_state, list_k, True,
        weights, validation_methods, baselines,
        cfg.default.mushroom_figures,
        'knn')

if True:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_knn(cfg.default.mushroom_figures, 'knn')
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_knn(cfg.default.mushroom_figures, 'knn')

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_knn(cfg.default.mushroom_figures, 'knn')
if True:
    # List variants with highest and lowest accuracy values
    path = cfg.default.occupancy_figures
    filename = 'knn'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())


#%% Naïve Bayes Classification
if False:
    # Scaling not needed for algorithms that don’t use distances like Naive
    # Bayes
    functions.gnb(X, y, test_size, random_state,
        validation_methods, baselines,
        cfg.default.occupancy_figures,
        'gnb')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_gnb(cfg.default.occupancy_figures, 'gnb')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_gnb(cfg.default.occupancy_figures, 'gnb')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_gnb(cfg.default.occupancy_figures, 'gnb')

#%% Decision Tree Classification
if True:
    list_max_depth = [1, 10, 100, 1000]
    list_min_samples_split = [2, 20, 200, 2000]
    list_min_samples_leaf = [1, 10, 200, 2000]

    functions.dt(X, y, test_size, random_state, list_max_depth,
        list_min_samples_split, list_min_samples_leaf,
        validation_methods, baselines,
        cfg.default.mushroom_figures,
        'dt')

if True:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_dt(cfg.default.mushroom_figures, 'dt')

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_dt(cfg.default.mushroom_figures, 'dt')
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_dt(cfg.default.mushroom_figures, 'dt')

if True:
    # List variants with highest and lowest accuracy values
    path = cfg.default.occupancy_figures
    filename = 'dt'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())

#%% Ridge Classification
if True:
    list_alpha = [0, 1e-4, 1e-2, 1, 5, 10, 50, 100]
    functions.ridge(X, y, test_size, random_state, list_alpha, True,
        validation_methods, baselines,
        cfg.default.mushroom_figures,
        'ridge')

if True:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_ridge(cfg.default.mushroom_figures, 'ridge')
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_ridge(cfg.default.mushroom_figures, 'ridge')

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_ridge(cfg.default.mushroom_figures, 'ridge')
if True:
    # List variants with highest and lowest accuracy values
    path = cfg.default.mushroom_figures
    filename = 'ridge'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())

#%% Compare the different classifiers 
filenames = ['knn', 'ridge', 'dt']
if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy(cfg.default.mushroom_figures, filenames)
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency(cfg.default.mushroom_figures, filenames)

if True:
    # List variants of each classifier with highest accuracy values
    all_evaluations = pd.DataFrame()
    for filename in filenames:
        path = cfg.default.mushroom_figures
        evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
            key='evaluation')

        # Select only rows with cross-validation and only Classifier (no baselines)
        rows = ()
        for name in evaluation.index.names:
            if name == 'validation method':
                rows = rows + ('cross-validation', )
            elif name == 'classifier':
                rows = rows + ('Classifier', )
            else:
                rows = rows + (slice(None), )
        evaluation = evaluation.loc[rows, ('accuracy MEAN', 'accuracy SD')]

        # Flatten multiIndex to tuple and add filename
        index = evaluation.index.to_flat_index()
        index_new = [' '.join([filename, str(entry)]) for entry in index]
        evaluation.index = index_new

        # Add current evaluation to overall comparison
        all_evaluations = pd.concat([all_evaluations, evaluation])

    print('Highest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].
        sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).head(10))
    print()
    print('Lowest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].
        sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[True, False])).head(10))

    # Save sorted DataFrame (descending mean value, ascending SD)
    # as csv and h5 files
    filename = '_'.join(['all_evaluation', '_'.join(filenames)])
    ((all_evaluations.sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).to_csv(os.path.join(path, filename + '.csv'), sep=';', decimal=','))
    ((evaluation.sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).to_hdf(os.path.join(path, filename + '.h5'), key='evaluation', mode='w'))



print()
print('Done')