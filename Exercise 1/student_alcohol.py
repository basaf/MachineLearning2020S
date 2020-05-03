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

rawData = pd.read_csv(os.path.join(cfg.default.student_data, 'student-mat.csv'), sep=';')

rawData['school'] = rawData['school'].astype('category')
rawData['school'] = rawData['school'].cat.codes

rawData['sex'] = rawData['sex'].astype('category')
rawData['sex'] = rawData['sex'].cat.codes

rawData['address'] = rawData['address'].astype('category')
rawData['address'] = rawData['address'].cat.codes

rawData['famsize'] = rawData['famsize'].astype('category')
rawData['famsize'] = rawData['famsize'].cat.codes

rawData['Pstatus'] = rawData['Pstatus'].astype('category')
rawData['Pstatus'] = rawData['Pstatus'].cat.codes

rawData['Mjob'] = rawData['Mjob'].astype('category')
rawData['Mjob'] = rawData['Mjob'].cat.codes

rawData['Fjob'] = rawData['Fjob'].astype('category')
rawData['Fjob'] = rawData['Fjob'].cat.codes

rawData['reason'] = rawData['reason'].astype('category')
rawData['reason'] = rawData['reason'].cat.codes

rawData['guardian'] = rawData['guardian'].astype('category')
rawData['guardian'] = rawData['guardian'].cat.codes

rawData['schoolsup'] = rawData['schoolsup'].astype('category')
rawData['schoolsup'] = rawData['schoolsup'].cat.codes

rawData['famsup'] = rawData['famsup'].astype('category')
rawData['famsup'] = rawData['famsup'].cat.codes

rawData['paid'] = rawData['paid'].astype('category')
rawData['paid'] = rawData['paid'].cat.codes

rawData['activities'] = rawData['activities'].astype('category')
rawData['activities'] = rawData['activities'].cat.codes

rawData['nursery'] = rawData['nursery'].astype('category')
rawData['nursery'] = rawData['nursery'].cat.codes

rawData['higher'] = rawData['higher'].astype('category')
rawData['higher'] = rawData['higher'].cat.codes

rawData['internet'] = rawData['internet'].astype('category')
rawData['internet'] = rawData['internet'].cat.codes

rawData['romantic'] = rawData['romantic'].astype('category')
rawData['romantic'] = rawData['romantic'].cat.codes

x = rawData.to_numpy()

data = rawData.copy()

X = data.drop(['G3'], axis=1)
Y = data['G3']

# %% investigate data
helper.boxplot_raw_data(rawData, X.columns, figsize=(15.00, 40.00),
                        save_fig_path=os.path.join(cfg.default.student_figures, 'student_box_plot.png'))

correlation_matrix = (X.corr(method='pearson'))

ax = heatmap(correlation_matrix, xticklabels=correlation_matrix.columns,
             yticklabels=correlation_matrix.columns, annot=False)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.student_figures, 'student_corr.png'), format='png')
plt.close()

figure = plt.figure()
Y.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.student_figures, 'student_target_hist.png'), format='png')
plt.close(figure)

X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

print('Ridge Linear Regression')

alpha_list = [0, .1, 0.3, .5, 1, 1.2]

functions.ridge_regression(X_train, X_test, Y_train, Y_test, alpha_list, True,
                           cfg.default.student_figures, 'ridge_reg')

print('KNN')

k_values = [1, 2, 5, 7, 10]

functions.knn(X_train, X_test, Y_train, Y_test, k_values, True, ['uniform', 'distance'],
              cfg.default.student_figures, 'knn')

print('Decission Tree Regression')

max_depths = [1, 10, 30, 50, 100, 300]
min_weight_fraction_leafs = [.0, .125, .25, .375, .5]
min_samples_leaf=[1, 10, 100, 200]

functions.decision_tree(X_train, X_test, Y_train, Y_test, max_depths, min_weight_fraction_leafs, min_samples_leaf,
                        cfg.default.student_figures, 'dtree')

print('MLP')
