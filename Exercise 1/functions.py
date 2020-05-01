# -*- coding: utf-8 -*-
"""
common functions for all datasets
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# damit man latex in den labels von den plots verwenden kann
matplotlib.rcParams['text.usetex'] = True


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


def mean_root_squared_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))) * 100


def check_performance(y_test, y_pred, filename=None):
    fig = plt.figure()
    plt.plot(y_test.values, label='y')
    plt.plot(y_pred, label='\^{y}')
    plt.legend()

    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    explained_variance_score = metrics.explained_variance_score(y_test, y_pred)

    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + ".png", format='png')

        f = open(filename + '.txt', 'w')
        f.write(f'Mean Absolute Error (MAE): {MAE:.2f}\n')
        f.write(f'Mean Absolute Percentage Error (MAPE): {MAPE:.2f}\n')
        f.write(f'Root Mean Squared Error (RMSE): {RMSE:.2f}\n')
        f.write(f'Explained Variance: {explained_variance_score:.2f}\n')
        f.close()

    print(f'Mean Absolute Error (MAE): {MAE:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {MAPE:.2f}')
    print(f'Root Mean Squared Error (RMSE): {RMSE:.2f}')
    print(f'Explained Variance: {explained_variance_score:.2f}')

    return fig, MAE, MSE, RMSE, explained_variance_score
