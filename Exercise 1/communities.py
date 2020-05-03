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
import seaborn as sns

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing

import functions
import helper

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
    alphas = [0, 0.5, 1, 5, 10, 50, 100]
    scalings = ['scaling', 'noScaling']
    index = pd.MultiIndex.from_product([alphas, scalings],
                                       names=['alpha', 'scaling'])
    RidgeRegression_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
    # Change parameters
    for alpha in alphas:
        for scaling in scalings:
            if scaling == 'scaling':
                xtrain = X_train_scaled
                xtest = X_test_scaled
                normalize = False
            else:  
                xtrain = X_train
                xtest = X_test
                normalize = True
            filename = 'RidgeRegression_'+str(alpha)+'_'+scaling
            
            reg = linear_model.Ridge(alpha=alpha, normalize=normalize)
            reg.fit(xtrain, y_train)
            y_pred_reg = reg.predict(xtest)

            errors = functions.check_performance(y_test, y_pred_reg,
                os.path.join(cfg.default.communities_figures, filename))
            RidgeRegression_errors.loc[alpha, scaling][:] = errors

            del xtrain, xtest
    
    print(RidgeRegression_errors)
    RidgeRegression_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures,
                     'RidgeRegression_errors.csv'),
        sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(RidgeRegression_errors.keys())):
        fig = plt.figure()
        ax = fig.add_subplot()
    for key in RidgeRegression_errors.keys():
        ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'scaling'),
                key].to_numpy(),
                marker='o', linestyle='-', label=key+' scaled')
    for key in RidgeRegression_errors.keys():
        ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'noScaling'),
                key].to_numpy(),
                marker='o', linestyle='--', label=key+' not scaled')
    plt.ylim([0, 1])
    plt.xlabel(r'$\alpha$')
    plt.grid()
    plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, -0.15))
    plt.show()
    fig.savefig(os.path.join(cfg.default.communities_figures,
                            'RidgeRegression_errors.png'),
                format='png', dpi=200, bbox_inches='tight',
                metadata={'Creator': '', 'Author': '', 'Title': '',
                        'Producer': ''},
                )
        

#%% k-Nearest Neighbor Regression
if False:
    list_k = [1, 3, 5, 10, 20, 50, 100, 300]
    scalings = ['scaling', 'noScaling']
    weights = ['uniform', 'distance']

    index = pd.MultiIndex.from_product([list_k, scalings, weights],
                                       names=['k', 'scaling', 'weights'])
    knn_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
    # Change parameters
    for k in list_k:
        for scaling in scalings:
            for weight in weights:
                if scaling == 'scaling':
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                else:  
                    xtrain = X_train
                    xtest = X_test
                filename = 'k-NN_'+str(k)+'_'+weight+'_'+scaling

                knn = KNeighborsRegressor(n_neighbors=k, weights=weight)
                knn.fit(xtrain, y_train)
                y_pred_knn = knn.predict(xtest)

                errors = functions.check_performance(y_test, y_pred_knn,
                    os.path.join(cfg.default.communities_figures, filename))
                knn_errors.loc[k, scaling, weight][:] = errors
                del xtrain, xtest

    print(knn_errors)
    knn_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'knn_errors.csv'),
        sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(knn_errors.keys())):
        fig = plt.figure()
        ax = fig.add_subplot()
    for key in knn_errors.keys():
        ax.plot(list_k, knn_errors.loc[(slice(None),'scaling', 'uniform'),
                key].to_numpy(),
                marker='o', linestyle='-', label=key+' scaled, unif')
    for key in knn_errors.keys():
        ax.plot(list_k, knn_errors.loc[(slice(None),'scaling', 'distance'),
                key].to_numpy(),
                marker='o', linestyle='-.', label=key+' scaled, dist')
    for key in knn_errors.keys():
        ax.plot(list_k, knn_errors.loc[(slice(None),'noScaling', 'uniform'),
                key].to_numpy(),
                marker='o', linestyle='--', label=key+' not scaled, unif')
    for key in knn_errors.keys():
        ax.plot(list_k, knn_errors.loc[(slice(None),'noScaling', 'distance'),
                key].to_numpy(),
                marker='o', linestyle=':', label=key+' not scaled, dist')
    plt.ylim([0, 1])
    plt.xlabel(r'$k$')
    plt.grid()
    plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, -0.15))
    plt.show()
    fig.savefig(os.path.join(cfg.default.communities_figures,
                            'knn_errors.png'),
                format='png', dpi=200, bbox_inches='tight',
                metadata={'Creator': '', 'Author': '', 'Title': '',
                        'Producer': ''},
                )

#%% Decision Tree Regression
if True:
    max_depths = [1, 10, 30, 50, 100, 300]  # , 500]
    min_weight_fraction_leafs = [.0, .125, .25, .375, .5]
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

    index = pd.MultiIndex.from_product([max_depths, min_weight_fraction_leafs],
                                       names=['max_depths',
                                              'min_weight_fraction_leaf'])
    dt_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

    for max_depth in max_depths:
        for min_weight_fraction_leaf in min_weight_fraction_leafs:
            xtrain = X_train
            xtest = X_test
            filename = 'DTRegressor_'+str(max_depth)+'_'+str(min_weight_fraction_leaf)

            dt = tree.DecisionTreeRegressor(max_depth=max_depth,
                min_weight_fraction_leaf=min_weight_fraction_leaf)
            dt.fit(xtrain, y_train)
            y_pred_dt = dt.predict(xtest)  

            errors = functions.check_performance(y_test, y_pred_dt,
                os.path.join(cfg.default.communities_figures, filename))
            dt_errors.loc[max_depth, min_weight_fraction_leaf][:] = errors
            del xtrain, xtest

    print(dt_errors)
    dt_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'dt_errors.csv'),
        sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(dt_errors.keys())):
        fig = plt.figure()
        ax = fig.add_subplot()
    linestyle_cycle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*']
    for idx, key2 in enumerate(min_weight_fraction_leafs):
        linestyle = linestyle_cycle[idx]
        marker = marker_cycle[idx]
        for key in dt_errors.keys():
            ax.plot(max_depths, dt_errors.loc[(slice(None), key2),
                    key].to_numpy(),
                    marker=marker, linestyle=linestyle, label=str(key)+', '+str(key2))
    plt.ylim([0, 1])
    plt.xlabel(r'$\mathrm{max depth}$')
    plt.grid()
    plt.legend(ncol=5, loc='upper left', bbox_to_anchor=(0, -0.15))
    plt.show()
    fig.savefig(os.path.join(cfg.default.communities_figures,
                            'dt_errors.png'),
                format='png', dpi=200, bbox_inches='tight',
                metadata={'Creator': '', 'Author': '', 'Title': '',
                        'Producer': ''},
                )



# %%
    stophere

    filename = 'DTRegressor'
    functions.check_performance(y_test, y_pred_dt)  # ,
        # os.path.join(cfg.default.communities_figures, filename))

print()
print('Done')


