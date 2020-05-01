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
    fig = plt.figure()
    plt.plot(y_test, label='y')
    plt.plot(y_pred, label=r'$\hat y$')
    plt.legend()

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
    
    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + ".png", format='png')

        f = open(filename + '.txt', 'w')
        f.write(f'Mean Absolute Error (MAE): {MAE:.2f}\n')
        f.write(f'Mean Absolute Percentage Error (MAPE): {MAPE:.2f}\n')
        f.write(f'Mean Squared Error (MSE): {:.2f}'.format(MSE))
        f.write(f'Root Mean Squared Error (RMSE): {RMSE:.2f}\n')
        f.write(f'Explained Variance (EV): {EV:.2f}\n')
        f.close()

    return fig, MAE, MSE, RMSE, EV
