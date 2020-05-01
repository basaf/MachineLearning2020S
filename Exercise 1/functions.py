# -*- coding: utf-8 -*-
"""
common functions for all datasets
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def plotPie(dataFrame):
    labels = dataFrame.astype('category').cat.categories.tolist()
    counts = dataFrame.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)* 100)

def mean_root_squared_percentage_error(y_true, y_pred): 
   y_true, y_pred = np.array(y_true), np.array(y_pred)
   return np.sqrt(np.mean(np.square((y_true - y_pred)/ y_true)))* 100 

def checkPerformance(y_test, y_pred):
    fig = plt.figure()
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='y_hat')
    plt.legend()
    plt.show()
    
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    # MAPE = mean_absolute_percentage_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # mean_root_squared_perc_err = mean_root_squared_percentage_error(y_test, y_pred)
    explained_variance_score = metrics.explained_variance_score(y_test, y_pred)

    print('Mean Absolute Error (MAE): {:.2f}'.format(MAE))
    # print('Mean Absolute Percentage Error (MAPE): {:.2f}'.format(MAPE))
    print('Mean Squared Error: {:.2f}'.format(mean_squared_error))
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(RMSE))
    # print('Root Relative Squared Error: {:.2f}'.format(mean_root_squared_perc_err))
    print('Explained Variance: {:.2f}'.format(explained_variance_score))

    return fig, MAE, mean_squared_error, RMSE, explained_variance_score
