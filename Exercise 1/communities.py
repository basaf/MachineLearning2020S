# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:31:00 2020

@author: Steindl, Windholz
"""
from arff import load
import configuration as cfg
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from seaborn import heatmap
import seaborn as sns

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing

import functions
import helper

#%% data pre-processing
# load dataset (.arff) into pandas DataFrame
rawData = load(open(os.path.join(cfg.default.communities_data,
                                 'communities.arff'), 'r'))
all_attributes = list(i[0] for i in rawData['attributes'])
communities_data = pd.DataFrame(columns=all_attributes, data=rawData['data'])

# distinguish attributes in not_predictive, predictive and goal
not_predictive_attributes = [
    'state',
    'county',
    'community',
    'communityname',
    'fold'
]
goal_attribute = 'ViolentCrimesPerPop'

predictive_attributes = all_attributes.copy()
predictive_attributes.remove(goal_attribute)

for x in not_predictive_attributes:
    predictive_attributes.remove(x)

#%% investigate data
if False:
    communities_data[predictive_attributes[0:30]].boxplot()
    communities_data[predictive_attributes[30:60]].boxplot()
    communities_data[predictive_attributes[60:90]].boxplot()

#%% Treat missing values
missing_values = (communities_data[predictive_attributes+[goal_attribute]].
                  isnull().sum().sum())
cells_total = (len(communities_data.index)*
    len(communities_data[predictive_attributes+[goal_attribute]].columns))
print('Missing values: '+str(missing_values))
print('Cells total: '+str(cells_total))
print('Missing: {:.1%}'.format(missing_values/cells_total))

# Remove attributes with more than 80 % missing values
attributes_to_delete = communities_data[predictive_attributes].columns[
    communities_data[predictive_attributes].isnull().sum() / 
    len(communities_data.index)*100 > 80]
for x in attributes_to_delete:
    predictive_attributes.remove(x)

print('Missing in "OtherPerCap": '+
      str(communities_data['OtherPerCap'].isnull().sum()))
# -> impute mean value of attribute, but do the split before

# Input variable correlation analysis
correlation_matrix = (communities_data[predictive_attributes+[goal_attribute]].
                      corr(method='pearson'))

if False:
    ax = heatmap(correlation_matrix, center=0, vmin=-1, vmax=1, square=True,
                xticklabels=False, yticklabels=False)
    # plt.gcf().subplots_adjust(bottom=0.48, left=0.27, right=0.99, top=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.communities_figures, 
                'communities_data_correlations.png'),
                format='png', dpi=200,
                metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                )

#%% Data encoding
# Not necessary 

#%% Split data
X = communities_data[predictive_attributes].to_numpy()
y = communities_data[goal_attribute].to_numpy() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

#%% Impute mean value of attributes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

#%% Data scaling (remove mean and scale to unit variance)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Ridge regression
if True:
    alpha_list = [0, 0.5, 1, 5, 10, 50, 100]
    functions.ridge_regression(X_train, X_test, y_train, y_test, alpha_list,
                               True, cfg.default.communities_figures,
                               'ridge_reg')

#%% k-Nearest Neighbor Regression
if True:
    k_values = [1, 3, 5, 10, 20, 50, 100, 300]
    functions.knn(X_train, X_test, y_train, y_test, k_values, True,
                  ['uniform', 'distance'], cfg.default.communities_figures,
                  'knn')

#%% Decision Tree Regression
if True:
    max_depths = [1, 10, 50, 100, 200, 500]
    min_samples_leaf = [1, 10, 100, 200]
    min_weight_fraction_leafs = [.0, .1, .2, .35, .5]

    functions.decision_tree(X_train, X_test, y_train, y_test, max_depths,
                            min_weight_fraction_leafs, min_samples_leaf,
                            cfg.default.communities_figures, 'dtree')

# %% Multi-layer Perceptron Regressor
if True:
    solver = 'lbfgs'  # default=’adam’
    # ‘adam’ works for large datasets (with thousands of training samples or more) in terms of both training time and validation score​
    # ‘lbfgs’ for small datasets can converge faster and perform better
    max_iteration = 800  # default=200
    alpha = [1e-7, 1e-4, 1e-1]  # usually in the range 10.0 ** -np.arange(1, 7)
    list_hidden_layer_sizes = [[90, 90], [180, 90, 180], [180, 180, 180]]


    functions.mlp(X_train_scaled, X_test_scaled, y_train, y_test, max_iteration, solver, alpha, list_hidden_layer_sizes,
            cfg.default.communities_figures, 'mlp')


#%% Finish
print()
print('Done')


