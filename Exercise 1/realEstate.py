# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

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

# change the transaction date to year, and month field
transactionDate = rawData['X1 transaction date']

transactionMonth = ((rawData['X1 transaction date'] - rawData['X1 transaction date'].astype(int)) / (1 / 12)).astype(
    int)
transactionYear = rawData['X1 transaction date'].astype(int)

# TODO: change Month to circular data

data = rawData.copy()
data.drop('X1 transaction date', axis=1, inplace=True)
data['X1 transaction year'] = transactionYear.values
data['X1 transaction month'] = transactionMonth.values

X = data.drop(['Y house price of unit area'], axis=1)
y = data['Y house price of unit area']

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

reg = linear_model.Ridge(alpha=.5)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

functions.check_performance(y_test, y_pred_reg, os.path.join(cfg.default.real_estate_figures, 'real_estate_ridge_linear'))

# will be normalized by subtracting mean and dividing by l2-norm
reg = linear_model.Ridge(alpha=.5, normalize=True)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

functions.check_performance(y_test, y_pred_reg, os.path.join(cfg.default.real_estate_figures, 'real_estate_ridge_linear_norm'))

print('KNN')

# without scaling
X_train_knn = X_train
X_test_knn = X_test

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # distance performs better
knn.fit(X_train_knn, y_train)
y_pred_knn = knn.predict(X_test_knn)

functions.check_performance(y_test, y_pred_knn, os.path.join(cfg.default.real_estate_figures, 'real_estate_knn'))

# with std scaler
X_train_knn = X_train_scaled_std
X_test_knn = X_test_scaled_std

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # distance performs better
knn.fit(X_train_knn, y_train)
y_pred_knn = knn.predict(X_test_knn)

functions.check_performance(y_test, y_pred_knn, os.path.join(cfg.default.real_estate_figures, 'real_estate_knn_std'))

# with min max scaler
X_train_knn = X_train_scaled_min_max
X_test_knn = X_test_scaled_min_max

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')  # distance performs better
knn.fit(X_train_knn, y_train)
y_pred_knn = knn.predict(X_test_knn)

functions.check_performance(y_test, y_pred_knn, os.path.join(cfg.default.real_estate_figures, 'real_estate_knn_min_max'))

print('Decission Tree Regression')

dt = tree.DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

functions.check_performance(y_test, y_pred_dt, os.path.join(cfg.default.real_estate_figures, 'real_estate_d_tree'))

print('Multi-layer Perceptron')
# without scaler
X_train_mlp = X_train
X_test_mlp = X_test
mlp = neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50, 10), max_iter=400, verbose=False)
mlp.fit(X_train_mlp, y_train)
y_pred_mlp = mlp.predict(X_test_mlp)

functions.check_performance(y_test, y_pred_mlp, os.path.join(cfg.default.real_estate_figures, 'real_estate_mlp'))

# with std scaler
X_train_mlp = X_train_scaled_std
X_test_mlp = X_test_scaled_std
mlp = neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50, 10), max_iter=400, verbose=False)
mlp.fit(X_train_mlp, y_train)
y_pred_mlp = mlp.predict(X_test_mlp)

functions.check_performance(y_test, y_pred_mlp, os.path.join(cfg.default.real_estate_figures, 'real_estate_mlp_std'))

# with min max scaler
X_train_mlp = X_train_scaled_min_max
X_test_mlp = X_test_scaled_min_max
mlp = neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50, 10), max_iter=400, verbose=False)
mlp.fit(X_train_mlp, y_train)
y_pred_mlp = mlp.predict(X_test_mlp)

functions.check_performance(y_test, y_pred_mlp, os.path.join(cfg.default.real_estate_figures, 'real_estate_mlp_min_max'))
