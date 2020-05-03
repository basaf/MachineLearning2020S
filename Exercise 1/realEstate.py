# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import configuration as cfg
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import helper
import functions

from seaborn import heatmap

rawData = pd.read_excel(os.path.join(cfg.default.real_estate_data, 'Real estate valuation data set.xlsx'))

# %% investigate data
helper.boxplot_raw_data(rawData, rawData.columns[[2, 3, 4, 5, 6, 7]],
                        save_fig_path=os.path.join(cfg.default.real_estate_figures, 'real_estate_box_plot.png'))

correlation_matrix = (rawData.loc[:, rawData.columns[[2, 3, 4, 5, 6, 7]]].corr(method='pearson'))

ax = heatmap(correlation_matrix, xticklabels=correlation_matrix.columns,
             yticklabels=correlation_matrix.columns, annot=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.real_estate_figures, 'real_estate_corr.png'), format='png')
plt.close()

figure = plt.figure()
rawData['Y house price of unit area'].hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.real_estate_figures, 'real_estate_target_hist.png'), format='png')
plt.close(figure)

# change the transaction date to year, and month field
transactionDate = rawData['X1 transaction date']

transactionMonth = \
    ((rawData['X1 transaction date'] - rawData['X1 transaction date'].astype(int)) / (1 / 12)).astype(int)
transactionYear = rawData['X1 transaction date'].astype(int)

data = rawData.copy()
data.drop('X1 transaction date', axis=1, inplace=True)
data['X1 transaction year'] = transactionYear.values
data['X1 transaction month'] = transactionMonth.values

X = data.drop(['Y house price of unit area'], axis=1).to_numpy()
y = data['Y house price of unit area'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print('Ridge Linear Regression')

alpha_list = [0, .1, 0.3, .5, 1, 1.2]

functions.ridge_regression(X_train, X_test, y_train, y_test, alpha_list, True,
                           cfg.default.real_estate_figures, 'ridge_reg')

print('KNN')

k_values = [1, 2, 5, 7, 10]

functions.knn(X_train, X_test, y_train, y_test, k_values, True, ['uniform', 'distance'],
              cfg.default.real_estate_figures, 'knn')

print('Decission Tree Regression')

max_depths = [1, 10, 30, 50, 100, 300]
min_weight_fraction_leafs = [.0, .125, .25, .375, .5]
min_samples_leaf=[1, 10, 100, 200]

functions.decision_tree(X_train, X_test, y_train, y_test, max_depths, min_weight_fraction_leafs, min_samples_leaf,
                        cfg.default.real_estate_figures, 'dtree')

print('MLP')

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

max_iteration = 1000
solver = 'lbfgs' # lbfgs, adam, sgd
alpha = [0.001, 0.0001, 0.00001]

list_hidden_layer_sizes = [[10], [5, 2, 5], [60, 20]]

functions.mlp(X_train_scaled, X_test_scaled, y_train, y_test, max_iteration, solver, alpha, list_hidden_layer_sizes,
        cfg.default.real_estate_figures, 'mlp')