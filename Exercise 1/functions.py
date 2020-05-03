# -*- coding: utf-8 -*-
"""
common functions for all datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn import neural_network
from sklearn import tree
import functions
import os
import seaborn as sns


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true) * 100)


def check_performance(y_test: np.array, y_pred: np.array, filename=None):
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
        plt.close()

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
        plt.close()

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

    if filename is not None:
        f = open(filename + '.txt', 'w')
        f.write(f'Mean Absolute Error (MAE): {MAE:.2f}\n')
        f.write(f'Mean Absolute Percentage Error (MAPE): {MAPE:.2f}\n')
        f.write(f'Mean Squared Error (MSE): {MSE:.2f}\n')
        f.write(f'Root Mean Squared Error (RMSE): {RMSE:.2f}\n')
        f.write(f'Explained Variance (EV): {EV:.2f}\n')
        f.close()

    return MAE, MAPE, MSE, RMSE, EV


def ridge_regression(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
                     alphas=[0, 0.5, 1, 5, 10, 50, 100], scaling: bool = True, path: str = None, filename: str = None):
    if scaling is True:
        scalings = ['scaling', 'noScaling']
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scalings = ['noScaling']

    index = pd.MultiIndex.from_product([alphas, scalings], names=['alpha', 'scaling'])
    errors = pd.DataFrame(index=index,
                          columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
    # Change parameters
    for alpha in alphas:
        for s in scalings:
            if s == 'scaling':
                xtrain = X_train_scaled
                xtest = X_test_scaled
                normalize = False
            else:
                xtrain = X_train
                xtest = X_test
                normalize = True

            reg = linear_model.Ridge(alpha=alpha, normalize=normalize)
            reg.fit(xtrain, Y_train)
            y_pred_reg = reg.predict(xtest)

            test_errors = functions.check_performance(Y_test, y_pred_reg,
                                                      os.path.join(path, filename + '_' + str(alpha) + '_' + s))
            errors.loc[alpha, s][:] = test_errors

            del xtrain, xtest

    print(errors)
    errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(errors.keys())):
        fig = plt.figure()

    for pos, key in enumerate(errors.keys()):
        ax = fig.add_subplot(5,1,pos+1)
        ax.set_title(key)
        if scaling is True:
            ax.plot(alphas, errors.loc[(slice(None), 'scaling'), key].to_numpy(),
                    marker='o', linestyle='-', label='scaled', alpha=0.8)
        ax.plot(alphas, errors.loc[(slice(None), 'noScaling'), key].to_numpy(),
                marker='o', linestyle='--', label='not scaled', alpha=0.8)
        
    # plt.ylim([0, 1])
    plt.xlabel(r'$\alpha$')
    plt.grid()
    plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, -0.15))
    # plt.show()
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename + '_errors.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def knn(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
        list_k=[1, 3, 5], scaling: bool = True, weights=['uniform', 'distance'],
        path: str = None, filename: str = None):
    if scaling is True:
        scalings = ['scaling', 'noScaling']
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scalings = ['noScaling']

    index = pd.MultiIndex.from_product([list_k, scalings, weights],
                                       names=['k', 'scaling', 'weights'])
    knn_errors = pd.DataFrame(index=index,
                              columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
    # Change parameters
    for k in list_k:
        for s in scalings:
            for weight in weights:
                if s == 'scaling':
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                else:
                    xtrain = X_train
                    xtest = X_test

                knn = KNeighborsRegressor(n_neighbors=k, weights=weight)
                knn.fit(xtrain, Y_train)
                y_pred_knn = knn.predict(xtest)

                errors = functions.check_performance(Y_test, y_pred_knn,
                                                     os.path.join(path,
                                                                  filename + '_' + str(k) + '_' + weight + '_' + s))
                knn_errors.loc[k, s, weight][:] = errors
                del xtrain, xtest

    print(knn_errors)
    knn_errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(knn_errors.keys())):
        fig = plt.figure()
       
    for pos, key in enumerate(knn_errors.keys()):
        ax = fig.add_subplot(5,1,pos+1)
        ax.set_title(key)
        
        ax.plot(list_k, knn_errors.loc[(slice(None), 'scaling', 'uniform'), key].to_numpy(),
                marker='o', linestyle='-', label='scaled, unif')

        ax.plot(list_k, knn_errors.loc[(slice(None), 'scaling', 'distance'), key].to_numpy(),
                marker='o', linestyle='-.', label='scaled, dist')

        ax.plot(list_k, knn_errors.loc[(slice(None), 'noScaling', 'uniform'), key].to_numpy(),
                marker='o', linestyle='--', label='not scaled, unif')

        ax.plot(list_k, knn_errors.loc[(slice(None), 'noScaling', 'distance'), key].to_numpy(),
                marker='o', linestyle=':', label='not scaled, dist')
        
    # plt.ylim([0, 1])
    plt.xlabel(r'$k$')
    plt.grid()
    plt.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, -0.15))
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename + '_errors.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def decision_tree(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
                  list_max_depth=[1, 10, 30, 50, 100, 300], list_min_weight_fraction_leaf=[.0, .125, .25, .375, .5],
                  list_min_samples_leaf=[1, 10, 100, 200], path: str = None, filename: str = None):
    index = pd.MultiIndex.from_product([list_max_depth,
                                        list_min_samples_leaf,
                                        list_min_weight_fraction_leaf],
                                       names=['max_depth', 'min_samples_leaf', 'min_weight_fraction_leaf'])
    dt_errors = pd.DataFrame(index=index, columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

    for max_depth in list_max_depth:
        for min_samples_leaf in list_min_samples_leaf:
            for min_weight_fraction_leaf in list_min_weight_fraction_leaf:
                xtrain = X_train
                xtest = X_test

                dt = tree.DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_weight_fraction_leaf=min_weight_fraction_leaf)
                dt.fit(xtrain, Y_train)
                y_pred_dt = dt.predict(xtest)

                errors = functions.check_performance(Y_test, y_pred_dt,
                                                     os.path.join(path,
                                                                  filename + '_' + str(max_depth) + '_' + str(
                                                                      min_samples_leaf) + '_' + str(
                                                                      min_weight_fraction_leaf)))
                dt_errors.loc[max_depth,
                              min_samples_leaf,
                              min_weight_fraction_leaf][:] = errors
                del xtrain, xtest

    print(dt_errors)
    dt_errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    for key3 in list_min_weight_fraction_leaf:
        with sns.color_palette(n_colors=len(dt_errors.keys())):
            fig = plt.figure()
            ax = fig.add_subplot()
        linestyle_cycle = ['-', '--', '-.', ':'] * 3  # to have enough elements (quick&dirty)
        marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*'] * 3  # to have enough elements (quick&dirty)
        for idx, key2 in enumerate(list_min_samples_leaf):
            linestyle = linestyle_cycle[idx]
            marker = marker_cycle[idx]
            for key in dt_errors.keys():
                ax.plot(list_max_depth, dt_errors.loc[(slice(None), key2, key3),
                                                      key].to_numpy(),
                        marker=marker, linestyle=linestyle, label=str(key) + ', ' + str(key2))
        # plt.ylim([0, 1])
        plt.xlabel('max_depth')
        plt.grid()
        plt.legend(title='min_samples_leaf', ncol=len(list_min_samples_leaf),
                   loc='upper left', bbox_to_anchor=(0, -0.15))
        plt.title('min_weight_fraction_leaf: ' + str(key3))
        fig.savefig(os.path.join(path, filename + '_errors_' + str(key3) + '.png'),
                    format='png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def mlp(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
        max_iter: int = 800, solver: str = 'lbfgs', list_alpha=[1e-7, 1e-4, 1e-1], list_hidden_layer_sizes = [[10]],
        path: str = None, filename: str = None): #list_neurons_per_hidden_layer, list_no_of_hidden_layers,

    index = pd.MultiIndex.from_product(
        [list_alpha, [str(x) for x in list_hidden_layer_sizes]],
        names=['alpha', 'hidden_layer_sizes'])
    mlp_errors = pd.DataFrame(index=index,
                              columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
    # Change parameters
    for alpha in list_alpha:
        for hidden_layer_sizes in list_hidden_layer_sizes:
            mlp = neural_network.MLPRegressor(solver=solver,
                                              max_iter=max_iter,
                                              alpha=alpha,
                                              hidden_layer_sizes=hidden_layer_sizes,
                                              verbose=True,
                                              random_state=5)

            mlp.fit(X_train, Y_train)
            y_pred_mlp = mlp.predict(X_test)

            errors = functions.check_performance(Y_test, y_pred_mlp,
                                                 os.path.join(path, filename + '_' + str(alpha) + '_' + str(
                                                     hidden_layer_sizes)))
            mlp_errors.loc[alpha, str(hidden_layer_sizes)][:] = errors

    print(mlp_errors)
    mlp_errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

    # Plot errors over parameters of algorithm
    with sns.color_palette(n_colors=len(mlp_errors.keys())):
        fig = plt.figure()

    linestyle_cycle = ['-', '--', '-.', ':'] * 3  # to have enough elements (quick&dirty)
    marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*'] * 3  # to have enough elements (quick&dirty)
    for idx, key2 in enumerate(list_hidden_layer_sizes):
        linestyle = linestyle_cycle[idx]
        marker = marker_cycle[idx]

        for pos,key in enumerate(mlp_errors.keys()):
            ax = fig.add_subplot(5,1,pos+1)
            ax.set_title(key)
            ax.semilogx(list_alpha, mlp_errors.loc[(slice(None), str(key2)), key].to_numpy(),
                        marker=marker, linestyle=linestyle,
                        label=str(key2))
            
    # plt.ylim([0, 1])
    plt.xlabel(r'$\alpha$')
    plt.grid()
    plt.legend(title='hidden_layer_sizes', ncol=len(list_hidden_layer_sizes), loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename + '_errors.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
