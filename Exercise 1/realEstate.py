# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

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

transactionMonth = ((rawData['X1 transaction date'] - rawData['X1 transaction date'].astype(int)) / (1 / 12)).astype(
    int)
transactionYear = rawData['X1 transaction date'].astype(int)

data = rawData.copy()
data.drop('X1 transaction date', axis=1, inplace=True)
data['X1 transaction year'] = transactionYear.values
data['X1 transaction month'] = transactionMonth.values

X = data.drop(['Y house price of unit area'], axis=1).to_numpy()
y = data['Y house price of unit area'].to_numpy()

# %%
# pd.scatter_matrix(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%Data scaling
scaler_std = preprocessing.StandardScaler().fit(X_train)
scaler_min_max = preprocessing.MinMaxScaler().fit(X_train)

X_train_scaled_std = scaler_std.transform(X_train)
X_test_scaled_std = scaler_std.transform(X_test)

X_train_scaled_min_max = scaler_min_max.transform(X_train)
X_test_scaled_min_max = scaler_min_max.transform(X_test)

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

functions.decision_tree(X_train, X_test, y_train, y_test, max_depths, min_weight_fraction_leafs,
                        cfg.default.real_estate_figures, 'dtree')

print('Random Forest')

X_train_rf = X_train
X_test_rf = X_test

n_values = [10, 30, 60, 100, 150]

for n in n_values:
    rf = RandomForestRegressor(n_estimators=n)

    rf.fit(X_train_rf, y_train)
    y_pred_rf = rf.predict(X_test_rf)

    functions.check_performance(y_test, y_pred_rf,
                                os.path.join(cfg.default.real_estate_figures, f'real_estate_rf_{str(n)}'))
