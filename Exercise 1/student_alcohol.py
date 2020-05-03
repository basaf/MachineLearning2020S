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
helper.boxplot_raw_data(rawData, X.columns,
                        save_fig_path=os.path.join(cfg.default.student_figures, 'student_box_plot.png'))

correlation_matrix = (X.corr(method='pearson'))

ax = heatmap(correlation_matrix, xticklabels=correlation_matrix.columns,
             yticklabels=correlation_matrix.columns, annot=True)
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print('Ridge Linear Regression')

print('KNN')

print('Decission Tree Regression')

print('Random Forest')
