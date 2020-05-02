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
if True:
    alphas = [0, 0.25, 0.5, 0.75, 1]
    scalings = ['scaling', 'noScaling']
    index = pd.MultiIndex.from_product([alphas, scalings],
                                       names=['alpha', 'scaling'])
    RidgeRegression_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

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
            filename = 'RidgeRegression_'+str(alpha)+scaling
            
            reg = linear_model.Ridge(alpha=alpha, normalize=normalize)
            reg.fit(xtrain, y_train)
            y_pred_reg = reg.predict(xtest)

            errors = functions.check_performance(y_test, y_pred_reg,
                os.path.join(cfg.default.communities_figures, filename))
            RidgeRegression_errors.loc[alpha, scaling][:] = errors

            del xtrain, xtest
    
    print(RidgeRegression_errors)
    RidgeRegression_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'RidgeRegression_errors.csv'),
        sep=';', decimal=',')

    # RidgeRegression_errors.plot(rot='90')

    # Variante 0
    # labels = [key for key in RidgeRegression_errors.loc[(slice(None),'scaling'),
    #                                     slice(None)].keys()]
    # ax = plt.gca()
    # ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'scaling'),
    #                                     slice(None)].to_numpy(),
    #         marker='o', linestyle='-', label=labels)
    # ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'noScaling'),
    #                                     slice(None)].to_numpy(),
    #         marker='o', linestyle='--')
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Variante 0.1
    plt.figure()
    ax = plt.gca()
    sns.set_palette(sns.color_palette('hls', 12))
    for key in RidgeRegression_errors.keys():
        ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'scaling'),
                                            key].to_numpy(),
                marker='o', linestyle='-', label=key+' scaled')
    sns.set_palette(sns.color_palette('hls', 12))
    for key in RidgeRegression_errors.keys():
        ax.plot(alphas, RidgeRegression_errors.loc[(slice(None),'noScaling'),
                                            key].to_numpy(),
                marker='o', linestyle='--', label=key+' not scaled')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Variante 1
    # ax = plt.gca()
    # RidgeRegression_errors.loc[(slice(None),'scaling'), slice(None)].plot(
    #     ax=ax,
    #     rot='90', marker='o', linestyle='-', grid=True)
    # RidgeRegression_errors.loc[(slice(None),'noScaling'), slice(None)].plot(
    #     ax=ax,
    #     rot='90', marker='o', linestyle='--', grid=True)        
    # plt.show()

    # # Variante 2
    # ax = plt.gca()
    # for key in RidgeRegression_errors.keys():
    #     RidgeRegression_errors.loc[(slice(None), slice(None)), key].plot(
    #         ax=ax,
    #         rot='90', marker='o', linestyle='--', grid=True)
    # # RidgeRegression_errors.loc[(slice(None),'noScaling'), slice(None)].plot(
    # #     ax=ax,
    # #     rot='90', marker='o', linestyle='-', grid=True)
    # plt.legend()        
    # plt.show()

    
    plt.savefig(os.path.join(cfg.default.communities_figures, 'RidgeRegression_errors.png'),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )        
        

#%% k-Nearest Neighbor Regression
if False:
    list_k = [1, 3, 5, 10, 100, 300]
    scalings = [True, False]
    weights = ['uniform', 'distance']

    index = pd.MultiIndex.from_product([list_k, scalings, weights],
                                       names=['k', 'scaling', 'weights'])
    knn_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

    for k in list_k:
        for scaling in scalings:
            for weight in weights:
                if scaling:
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                    normalize = False
                    filename = 'k-NN_'+str(k)+'_'+weight+'_scaling'
                else:  
                    xtrain = X_train
                    xtest = X_test
                    normalize = True
                    filename = 'k-NN_'+str(k)+'_'+weight+'_noScaling'

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
    knn_errors.plot(rot='90')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.communities_figures, 'knn_errors.png'),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )


#%% Decision Tree Regression
if False:

    stoppedhere

    max_depths = [1, 5, 10, 100, 300]
    min_samples_leaf = [1, 5, 10, 100, 300]
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

    index = pd.MultiIndex.from_product([max_depths, weights],
                                       names=['max_depths', 'weights'])
    knn_errors = pd.DataFrame(index=index,
        columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

    for max_depth in max_depths:
        for scaling in scalings:
            for weight in weights:
                if scaling:
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                    normalize = False
                    filename = 'DTRegressor_'+str(max_depth)+'_'+weight+'_scaling'
                else:  
                    xtrain = X_train
                    xtest = X_test
                    normalize = True
                    filename = 'DTRegressor_'+str(max_depth)+'_'+weight+'_noScaling'

                dt = tree.DecisionTreeRegressor(max_depth=max_depth) #MSE for measuring the quality of the split 
                dt.fit(X_train,y_train)
                y_pred_dt = dt.predict(X_test)  

                errors = functions.check_performance(y_test, y_pred_dt,
                    os.path.join(cfg.default.communities_figures, filename))
                dt_errors.loc[max_depth, scaling, weight][:] = errors
                del xtrain, xtest

    print(dt_errors)
    dt_errors.transpose().to_csv(
        os.path.join(cfg.default.communities_figures, 'dt_errors.csv'),
        sep=';', decimal=',')
    dt_errors.plot(rot='90')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.communities_figures, 'dt_errors.png'),
                            format='png', dpi=200,
                            metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                            )







    filename = 'DTRegressor'
    functions.check_performance(y_test, y_pred_dt)  # ,
        # os.path.join(cfg.default.communities_figures, filename))

print()
print('Done')



# %%
