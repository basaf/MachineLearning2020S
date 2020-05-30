# -*- coding: utf-8 -*-

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

#%% data pre-processing

occupancy_TrainingData = pd.read_csv(
    os.path.join(cfg.default.occupancy_data, 'datatraining.txt'))
occupancy_TrainingData = occupancy_TrainingData.set_index('date')
# occupancy_TrainingData['HumidityRatio g/kg'] = (
#     occupancy_TrainingData['HumidityRatio']*1000 )

occupancy_test_data = pd.read_csv(
    cfg.default.occupancy_data + '\\datatest.txt')
occupancy_test_data = occupancy_test_data.set_index('date')
# occupancy_test_data['HumidityRatio g/kg'] = (
#     occupancy_test_data['HumidityRatio']*1000 )

occupancy_test2_data = pd.read_csv(
    cfg.default.occupancy_data + '\\datatest2.txt')
occupancy_test2_data = occupancy_test2_data.set_index('date')
# occupancy_test2_data['HumidityRatio g/kg'] = (
#     occupancy_test2_data['HumidityRatio']*1000 )

# Concatenate data sets
data = pd.concat([occupancy_TrainingData,
    occupancy_test_data,
    occupancy_test2_data],
    sort=True, verify_integrity=True)

# distinguish attributes in predictive and goal
goal_attribute = 'Occupancy'

predictive_attributes = data.columns.to_list()
predictive_attributes.remove(goal_attribute)

#%% Investigate data
if False:
    helper.boxplot_raw_data(data,
        data[predictive_attributes].columns,
        save_fig_path=os.path.join(cfg.default.occupancy_figures,
            'occupancy_box_plot.png'))

#%% Treat missing values
missing_values = (data[predictive_attributes+[goal_attribute]].
                  isnull().sum().sum())
cells_total = (len(data.index)*
    len(data[predictive_attributes+[goal_attribute]].columns))
print('Missing values: '+str(missing_values))
print('Cells total: '+str(cells_total))
print('Missing: {:.1%}'.format(missing_values/cells_total))

# Remove attributes with more than 80 % missing values
if False:
    attributes_to_delete = data[predictive_attributes].columns[
        data[predictive_attributes].isnull().sum() / 
        len(data.index)*100 > 80]
    for x in attributes_to_delete:
        predictive_attributes.remove(x)

#%% Data encoding
# use month and weekday and hour of day as input with simple label encoding
data['dayOfWeek'] = pd.to_datetime(data.index).dayofweek
data['hourOfDay'] = pd.to_datetime(data.index).hour
predictive_attributes.append('dayOfWeek')
predictive_attributes.append('hourOfDay')

#%% Variable correlation analysis
correlation_matrix = (data[predictive_attributes+[goal_attribute]].
                      corr(method='pearson'))
if False:
    ax = heatmap(correlation_matrix, center=0, vmin=-1, vmax=1, square=True, 
        xticklabels=True, yticklabels=True, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 
        horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.occupancy_figures,
        'occupancy_data_correlations.png'),
        format='png', dpi=200,
        metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})

#%% Split data
X = data[predictive_attributes].to_numpy()
y = data[goal_attribute].to_numpy() 
test_size = 0.2
random_state = 1
# Splitting is done in sub routines, since whole data set is used for k-fold CV
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#     random_state=1)

#%% Impute mean value of attributes
if False:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

#%% Data scaling (remove mean and scale to unit variance)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

#%% k-Nearest Neighbor Classification
if False:
    list_k = [1, 10, 50, 100, 300, 500]
    functions.knn(X, y, test_size, random_state, list_k, True,
        ['uniform', 'distance'],
        ['holdout', 'cross-validation'],
        ['stratified', 'uniform'],
        cfg.default.occupancy_figures,
        'knn')

if True:
    functions.plot_evaluation_knn(cfg.default.occupancy_figures, 'knn')

#%% Naïve Bayes Classification
if False:
    # Scaling not needed for algorithms that don’t use distances like Naive
    # Bayes

    functions.gnb(X, y, test_size, random_state,
        ['holdout', 'cross-validation'],
        cfg.default.occupancy_figures,
        'gnb')

#%% Decision Tree Regression
# if False:
#     max_depths = [1, 10, 50, 100, 200, 500]
#     min_samples_leaf = [1, 10, 100, 200]
#     min_weight_fraction_leafs = [.0, .1, .2, .35, .5]

#     functions.decision_tree(X_train, X_test, y_train, y_test, max_depths,
#                             min_weight_fraction_leafs, min_samples_leaf,
#                             cfg.default.occupancy_figures, 'dtree')

# # %% Multi-layer Perceptron Regressor
# if False:
#     solver = 'lbfgs'  # default=’adam’
#     # ‘adam’ works for large datasets (with thousands of training samples or more) in terms of both training time and validation score​
#     # ‘lbfgs’ for small datasets can converge faster and perform better
#     max_iteration = 800  # default=200
#     alpha = [1e-7, 1e-4, 1e-1]  # usually in the range 10.0 ** -np.arange(1, 7)
#     list_hidden_layer_sizes = [[90, 90], [180, 90, 180], [180, 180, 180]]

#     functions.mlp(X_train_scaled, X_test_scaled, y_train, y_test, max_iteration, solver, alpha, list_hidden_layer_sizes,
#             cfg.default.occupancy_figures, 'mlp')


#%% Finish
print()
print('Done')


