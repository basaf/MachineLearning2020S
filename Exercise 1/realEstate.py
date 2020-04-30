# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import configuration as cfg
import os
import math

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing


def plotPie(dataFrame):
    labels = dataFrame.astype('category').cat.categories.tolist()
    counts = dataFrame.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)  # autopct is show the % on plot
    ax1.axis('equal')
    plt.show()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)


# def mean_root_squared_percentage_error(y_true, y_pred):
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.sqrt(np.mean(np.square((y_true - y_pred)/ y_true)))* 100

def checkPerformance(y_test, y_pred, filename=None):
    plt.figure()
    plt.plot(y_test.values, label='true')
    plt.plot(y_pred, label='y_hat')
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename, format='png')

    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # print('Root Relative Squared Error:', mean_root_squared_percentage_error(y_test, y_pred))
    print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))


rawData = pd.read_excel(os.path.join(cfg.default.real_estate_data, 'Real estate valuation data set.xlsx'))

# %% investigate data

# change the transaction date to year, and month field
transactionDate = rawData['X1 transaction date']

transactionMonth = ((rawData['X1 transaction date']- rawData['X1 transaction date'].astype(int))/(1/12)).astype(int)
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

#%%Data scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

print('Ridge Linear Regression')

#will be normalized by subtracting mean and dividing by l2-norm
reg = linear_model.Ridge(alpha=.5)#,normalize=True)
reg.fit(X_train,y_train)
y_pred_reg=reg.predict(X_test)

checkPerformance(y_test, y_pred_reg, os.path.join(cfg.default.real_estate_figures, 'real_estate_ridge_linear.png'))

print('KNN')
#scaling - makes the reults worse!!??

X_train_knn=X_train_scaled
X_test_knn=X_test_scaled
#Without scaling:
#X_train_knn=X_train
#X_test_knn=X_test

knn = KNeighborsRegressor(n_neighbors=5, weights='distance') #distance performs better
knn.fit(X_train_knn,y_train)
y_pred_knn=knn.predict(X_test_knn)

checkPerformance(y_test, y_pred_knn, os.path.join(cfg.default.real_estate_figures, 'real_estate_knn.png'))

print('Decission Tree Regression')

dt = tree.DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)

checkPerformance(y_test, y_pred_dt, os.path.join(cfg.default.real_estate_figures, 'real_estate_d_tree.png'))

print('Multi-layer Perceptron')

X_train_mlp=X_train_scaled
X_test_mlp=X_test_scaled
mlp=neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50,10), max_iter=400, verbose=False)
mlp.fit(X_train_mlp,y_train)
y_pred_mlp=mlp.predict(X_test_mlp)

checkPerformance(y_test, y_pred_mlp, os.path.join(cfg.default.real_estate_figures, 'real_estate_mlp.png'))
