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

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing

import functions

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
    plt.savefig(os.path.join(cfg.default.dataset_communities_figures_path, 
                'communities_data_correlations.png'),
                format='png', dpi=200,
                metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                )

#%% Data encoding
# Not necessary 

#%% Split data
X = communities_data[predictive_attributes].to_numpy()
y = communities_data[goal_attribute].to_numpy() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Impute mean value of attributes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

#%% Data scaling (remove mean and scale to unit variance)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Ridge regression
reg = linear_model.Ridge(alpha=0.5)
reg.fit(X_train_scaled, y_train)
y_pred_reg = reg.predict(X_test_scaled)
res = functions.checkPerformance(y_test, y_pred_reg)
fig, errors = res[0], res[1:]

plt.figure = fig
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.communities_figures, 
            'RidgeRegression_0.5_scaling.png'),
            format='png', dpi=200,
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
            )
stophere

# The same without scaling
reg = linear_model.Ridge(alpha=0.5, normalize=True)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
res = functions.checkPerformance(y_test, y_pred_reg)
fig, errors = res[0], res[1:]

plt.tight_layout()
plt.savefig(os.path.join(cfg.default.communities_figures, 
            'RidgeRegression_0.5_noScaling.png'),
            format='png', dpi=200,
            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
            )


stophere
#%%KNN
#scaling - makes the reults worse!!??

X_train_knn=X_train_scaled
X_test_knn=X_test_scaled
#Without scaling:
#X_train_knn=X_train
#X_test_knn=X_test

knn = KNeighborsRegressor(n_neighbors=5, weights='distance') #distance performs better
knn.fit(X_train_knn,y_train)
y_pred_knn=knn.predict(X_test_knn)

functions.checkPerformance(y_test, y_pred_knn)

#%%Decission Tree Regression

dt = tree.DecisionTreeRegressor() #MSE for measuring the quality of the split 
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)

functions.checkPerformance(y_test, y_pred_dt)

#%%Multi-layer Perceptron
X_train_mlp=X_train_scaled
X_test_mlp=X_test_scaled
mlp=neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50,10), max_iter=400,verbose=True)
mlp.fit(X_train_mlp,y_train)
y_pred_mlp=mlp.predict(X_test_mlp)

functions.checkPerformance(y_test, y_pred_mlp)