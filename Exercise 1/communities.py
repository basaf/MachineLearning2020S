# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:31:00 2020

@author: Steindl, Windholz
"""
from arff import load
import configuration as cfg
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from seaborn import heatmap

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing

import functions

#%% data pre-processing
# load dataset (.arff) into pandas DataFrame
rawData = load(open(os.path.join(cfg.default.communities_data,
                                 'communities.arff'), 'r'))
all_attributes = list(i[0] for i in rawData['attributes'])
communities_data = pd.DataFrame(columns=all_attributes, data=rawData['data'])

# distinguish attributes in not_predictive, predictive and goal
not_predictive_attributes = [
    'state',
    'county',
    'community',
    'communityname',
    'fold'
]
goal_attribute = 'ViolentCrimesPerPop'

predictive_attributes = all_attributes.copy()
predictive_attributes.remove(goal_attribute)

for x in not_predictive_attributes:
    predictive_attributes.remove(x)

#%% investigate data
if False:
    communities_data[predictive_attributes[0:30]].boxplot()
    communities_data[predictive_attributes[30:60]].boxplot()
    communities_data[predictive_attributes[60:90]].boxplot()

#%% Treat missing values
missing_values = (communities_data[predictive_attributes+[goal_attribute]].
                  isnull().sum().sum())
cells_total = (len(communities_data.index)*
    len(communities_data[predictive_attributes+[goal_attribute]].columns))
print('Missing values: '+str(missing_values))
print('Cells total: '+str(cells_total))
print('Missing: {:.1%}'.format(missing_values/cells_total))

# Remove attributes with more than 80 % missing values
attributes_to_delete = communities_data[predictive_attributes].columns[
    communities_data[predictive_attributes].isnull().sum() / 
    len(communities_data.index)*100 > 80]
for x in attributes_to_delete:
    predictive_attributes.remove(x)

print('Missing in "OtherPerCap": '+
      str(communities_data['OtherPerCap'].isnull().sum()))
# -> impute mean value of attribute, but do the split before

# Input variable correlation analysis
correlation_matrix = (communities_data[predictive_attributes+[goal_attribute]].
                      corr(method='pearson'))

if False:
    ax = heatmap(correlation_matrix, center=0, vmin=-1, vmax=1, square=True,
                xticklabels=False, yticklabels=False)
    # plt.gcf().subplots_adjust(bottom=0.48, left=0.27, right=0.99, top=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.dataset_communities_figures_path, 
                'communities_data_correlations.png'),
                format='png', dpi=200,
                metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                )

#%% Data encoding
# Not necessary 

#%% Split data
X = communities_data[predictive_attributes].to_numpy()
y = communities_data[goal_attribute].to_numpy() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Impute mean value of attributes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

#%% Data scaling (remove mean and scale to unit variance)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Ridge regression
if False:
    alphas = [0, 0.25, 0.5, 0.75, 1]
    scalings = [True, False]

    index = pd.MultiIndex.from_product([alphas, scalings], names=['alpha', 'scaling'])
    RidgeRegression_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MSE', 'RMSE', 'EV'])

    for alpha in alphas:
        for scaling in scalings:
            if scaling:
                xtrain = X_train_scaled
                xtest = X_test_scaled
                normalize = False
                filename = 'RidgeRegression_'+str(alpha)+'_scaling.png'
            else:  
                xtrain = X_train
                xtest = X_test
                normalize = True
                filename = 'RidgeRegression_'+str(alpha)+'_noScaling.png'
            
            reg = linear_model.Ridge(alpha=alpha, normalize=normalize)
            reg.fit(xtrain, y_train)
            y_pred_reg = reg.predict(xtest)
            res = functions.check_performance(y_test, y_pred_reg)
            fig, errors = res[0], res[1:]

            fig.tight_layout()
            fig.savefig(os.path.join(cfg.default.communities_figures, filename),
                        format='png', dpi=200,
                        metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                        )

            RidgeRegression_errors.loc[alpha, scaling][:] = errors
            del xtrain, xtest

    print(RidgeRegression_errors)
    RidgeRegression_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'RidgeRegression_errors.csv'),
        sep=';', decimal=',')
    RidgeRegression_errors.plot(rot='90')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.communities_figures, 'RidgeRegression_errors.png'),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )        
        

#%% k-Nearest Neighbor Regression
if False:
    list_k = [1, 3, 5, 10, 100, 300]
    scalings = [True, False]
    weights = ['uniform', 'distance']

    index = pd.MultiIndex.from_product([list_k, scalings, weights], names=['k', 'scaling', 'weights'])
    knn_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MSE', 'RMSE', 'EV'])

    for k in list_k:
        for scaling in scalings:
            for weight in weights:
                if scaling:
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                    normalize = False
                    filename = 'k-NN_'+str(k)+'_'+weight+'_scaling.png'
                else:  
                    xtrain = X_train
                    xtest = X_test
                    normalize = True
                    filename = 'k-NN_'+str(k)+'_'+weight+'_noScaling.png'

                knn = KNeighborsRegressor(n_neighbors=k, weights=weight)
                knn.fit(xtrain, y_train)
                y_pred_knn = knn.predict(xtest)

                res = functions.checkPerformance(y_test, y_pred_knn)
                fig, errors = res[0], res[1:]

                fig.tight_layout()
                fig.savefig(os.path.join(cfg.default.communities_figures, filename),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )

                knn_errors.loc[k, scaling, weight][:] = errors
                del xtrain, xtest

    print(knn_errors)
    knn_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'knn_errors.csv'),
        sep=';', decimal=',')
    knn_errors.plot(rot='90')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.communities_figures, 'knn_errors.png'),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )


#%% Decision Tree Regression
if False:
    dt = tree.DecisionTreeRegressor() #MSE for measuring the quality of the split 
    dt.fit(X_train,y_train)
    y_pred_dt=dt.predict(X_test)

    functions.checkPerformance(y_test, y_pred_dt)

print()
print('Done')

