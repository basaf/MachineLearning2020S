# -*- coding: utf-8 -*-
"""
common functions for all datasets
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)

def check_performance(y_test, y_pred, filename=None):
    # Line plot
    plt.figure()
    plt.plot(y_test, label=r'$y$')
    plt.plot(y_pred, label=r'$\hat y$')
    plt.grid()
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + ".png", format='png')

    # Scatter plot
    plt.figure()
    plt.scatter(y_test, y_test, label=r'$y$', alpha=0.6)
    plt.scatter(y_test, y_pred, label=r'$\hat y$', alpha=0.6)
    plt.grid()
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + "_scatter.png", format='png')

    # Error values
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    EV = metrics.explained_variance_score(y_test, y_pred)

    print('Mean Absolute Error (MAE): {:.2f}'.format(MAE))
    print('Mean Absolute Percentage Error (MAPE): {:.2f}'.format(MAPE))
    print('Mean Squared Error (MSE): {:.2f}'.format(MSE))
    print('Root Mean Squared Error (RMSE): {:.2f}'.format(RMSE))
    print('Explained Variance (EV): {:.2f}'.format(EV))
    print()
    
    if filename is None:
        pass
    else:
        f = open(filename + '.txt', 'w')
        f.write(f'Mean Absolute Error (MAE): {MAE:.2f}\n')
        f.write(f'Mean Absolute Percentage Error (MAPE): {MAPE:.2f}\n')
        f.write(f'Mean Squared Error (MSE): {MSE:.2f}\n')
        f.write(f'Root Mean Squared Error (RMSE): {RMSE:.2f}\n')
        f.write(f'Explained Variance (EV): {EV:.2f}\n')
        f.close()

    return MAE, MAPE, MSE, RMSE, EV
