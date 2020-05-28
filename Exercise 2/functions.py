# -*- coding: utf-8 -*-
"""
common functions for all datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

from sklearn import metrics, linear_model
from sklearn import preprocessing
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn import neural_network
# from sklearn import tree

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix   # not available in 0.21.3
from helper import plot_confusion_matrix  # source copied from newest sklearn
from sklearn.metrics import classification_report

from sklearn.dummy import DummyClassifier

import functions
import os
import seaborn as sns

from time import process_time


def check_performance_holdout(y_test: np.array, y_pred: np.array,
                              filename=None):
    # Evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)
    binary_classification = ('accuracy' in dict_report.keys())

    # Generate columns for result
    if binary_classification:  # Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise.
        scores = ['accuracy', 'precision', 'recall', 'f1-score']
        averages = ['', 'macro avg', 'macro avg', 'macro avg']
        index_elements = list(zip(scores, averages))
    else:
        scores = ['precision', 'recall', 'f1-score']
        averages = ['micro avg', 'macro avg']
        index_elements = list(product(scores, averages))
        
    # index = [' '.join([score, average]).strip()
    #     for score, average in index_elements]

    scoring = [' '.join([score, average]).strip().
        replace(' avg', '').replace(' ', '_').replace('-', '_').
        replace('f1_score', 'f1')
        for score, average in index_elements]
    
    index = []
    # Add mean and standard deviations to index
    for score in scoring:
        index.append(score+' MEAN')
        index.append(score+' SD')
    # # Add mean values and standard deviation of fit and score times to index 
    # for time in ['fit', 'score']:
    #     for stat in ['MEAN', 'SD']:
    #         index.append(time+' time '+stat)
    result = pd.DataFrame(index=index, columns=['value'])

    # Pick results from report
    for idx, (score, average) in enumerate(index_elements):
        # Mean value 
        if score == 'accuracy':
            result.loc[index[idx*2]]['value'] = dict_report[score]
        else: 
            result.loc[index[idx*2]]['value'] = dict_report[average][score]
        # Standard deviation
        result.loc[index[idx*2+1]]['value'] = 0

    print(result)
    print()

    if filename is not None:
        result.to_csv(filename+'.txt', float_format='%.2f', sep='\t',
            header=False)

    return result


def check_performance_CV(classifier, X:np.array, y:np.array,
    cv=5, n_jobs=-1, filename=None):

    # Detect, wether it is a binary classification
    binary_classification = (len(np.unique(y)) == 2)

    # Generate columns for result
    if binary_classification:  # Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise.
        scores = ['accuracy', 'precision', 'recall', 'f1-score']
        averages = ['', 'macro avg', 'macro avg', 'macro avg']
        index_elements = list(zip(scores, averages))
    else:
        scores = ['precision', 'recall', 'f1-score']
        averages = ['micro avg', 'macro avg']
        index_elements = list(product(scores, averages))

    scoring = [' '.join([score, average]).strip().
        replace(' avg', '').replace(' ', '_').replace('-', '_').
        replace('f1_score', 'f1')
        for score, average in index_elements]
    

    index = []
    # Add mean and standard deviations to index
    for score in scoring:
        index.append(score+' MEAN')
        index.append(score+' SD')
    # Add mean values and standard deviation of fit and score times to index 
    for time in ['fit', 'score']:
        for stat in ['MEAN', 'SD']:
            index.append(time+' time '+stat)
    result = pd.DataFrame(index=index, columns=['value'])

    dict_CV_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring,
                                     n_jobs=n_jobs)

    # Pick results from cross_validate
    for idx in range(len(index_elements)):
        # Mean value 
        result.loc[index[idx*2]]['value'] = np.mean(
            dict_CV_results['test_'+scoring[idx]])
        # Standard deviation
        result.loc[index[idx*2+1]]['value'] = np.std(
            dict_CV_results['test_'+scoring[idx]])

    # Pick fit_time and score_time from cross_validate
    for time in ['fit', 'score']:
        # Mean value 
        result.loc[time+' time MEAN']['value'] = np.mean(
            dict_CV_results[time+'_time'])
        # Standard deviation
        result.loc[time+' time SD']['value'] = np.std(
            dict_CV_results[time+'_time'])
    
    print(result)
    print()

    if filename is not None:
        result.to_csv(filename+'.txt', float_format='%.2f', sep='\t',
            header=False)

    return result


# def ridge_regression(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
#                      alphas=[0, 0.5, 1, 5, 10, 50, 100], scaling: bool = True, path: str = None, filename: str = None):
#     if scaling is True:
#         scalings = ['scaling', 'noScaling']
#         scaler = preprocessing.StandardScaler().fit(X_train)
#         X_train_scaled = scaler.transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#     else:
#         scalings = ['noScaling']

#     index = pd.MultiIndex.from_product([alphas, scalings], names=['alpha', 'scaling'])
#     errors = pd.DataFrame(index=index,
#                           columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
#     # Change parameters
#     for alpha in alphas:
#         for s in scalings:
#             if s == 'scaling':
#                 xtrain = X_train_scaled
#                 xtest = X_test_scaled
#                 normalize = False
#             else:
#                 xtrain = X_train
#                 xtest = X_test
#                 normalize = True

#             reg = linear_model.Ridge(alpha=alpha, normalize=normalize)
#             reg.fit(xtrain, Y_train)
#             y_pred_reg = reg.predict(xtest)

#             test_errors = functions.check_performance_holdout(Y_test, y_pred_reg,
#                                                       os.path.join(path, filename + '_' + str(alpha) + '_' + s))
#             errors.loc[alpha, s][:] = test_errors

#             del xtrain, xtest

#     print(errors)
#     errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

#     # Plot errors over parameters of algorithm
#     with sns.color_palette(n_colors=len(errors.keys())):
#         #fig = plt.figure()
#         fig, ax = plt.subplots(len(errors.keys()), 1, sharex=True, tight_layout=True)

#     for pos, key in enumerate(errors.keys()):
#         #ax = fig.add_subplot(5, 1, pos + 1)
#         ax[pos].set_title(key)
#         ax[pos].grid(True)

#         if scaling is True:
#             ax[pos].plot(alphas, errors.loc[(slice(None), 'scaling'), key].to_numpy(),
#                          marker='o', linestyle='-', label='scaled', alpha=0.8)
#         ax[pos].plot(alphas, errors.loc[(slice(None), 'noScaling'), key].to_numpy(),
#                      marker='o', linestyle='--', label='not scaled', alpha=0.8)

#     plt.subplots_adjust(hspace=2.2)
#     plt.xlabel(r'$\alpha$')
#     plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.7))
#     fig.savefig(os.path.join(path, filename + '_errors.png'), format='png',
#                 dpi=200, bbox_inches='tight')
#     plt.close(fig)


def knn(X: np.array, y: np.array, test_size, random_state,
        list_k=[1, 3, 5], scaling: bool = True,
        weights=['uniform', 'distance'],
        validation_methods=['holdout', 'cross-validation'],
        baselines=['stratified', 'uniform'],
        path: str = None, filename: str = None):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state)

    if scaling is True:
        scalings = ['scaling', 'noScaling']
        # Data scaling (remove mean and scale to unit variance)        

        # For holdout
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # For k-fold CV
        X_scaled = preprocessing.StandardScaler().fit_transform(X)
    else:
        scalings = ['noScaling']

    index = pd.MultiIndex.from_product([list_k, scalings, weights,
        validation_methods,
        ['Classifier']+['Baseline '+i for i in baselines]],
        names=['k', 'scaling', 'weights', 'validation method', 'classifier'])

    evaluation = pd.DataFrame(index=index)
    # Add empty entries for fit and score times (mean, SD)
    for entry1 in ['fit', 'score']:
        for entry2 in ['MEAN', 'SD']:
            entry = ' '.join([entry1, 'time', entry2])
            if entry not in evaluation:
                evaluation[entry] = np.nan
                
    # Change parameters
    for k in list_k:
        for s in scalings:
            for weight in weights:
                if s == 'scaling':
                    # For holdout
                    xtrain = X_train_scaled
                    xtest = X_test_scaled
                    # For k-fold CV
                    x = X_scaled
                else:
                    # For holdout
                    xtrain = X_train
                    xtest = X_test
                    # For k-fold CV
                    x = X

                for method in validation_methods:
                    knn = KNeighborsClassifier(n_neighbors=k,
                                               weights=weight,
                                               algorithm='auto',  # 'ball_tree', 'kd_tree', 'brute'
                                               leaf_size=30,  # default=30
                                               n_jobs=-1)
                    if method == 'holdout':

                        tic = process_time()
                        knn.fit(xtrain, y_train)
                        fit_time = process_time()-tic

                        tic = process_time()
                        y_pred_knn = knn.predict(xtest)
                        performance = (
                            check_performance_holdout(y_test, y_pred_knn,
                                os.path.join(path,
                                    '_'.join([filename, str(k), weight, s,
                                        method])))
                            )
                        score_time = process_time()-tic

                        # Write fit and score times
                        evaluation.loc[k, s, weight, method, 'Classifier'][
                            'fit time MEAN'] = fit_time
                        evaluation.loc[k, s, weight, method, 'Classifier'][
                            'fit time SD'] = 0
                        evaluation.loc[k, s, weight, method, 'Classifier'][
                            'score time MEAN'] = score_time
                        evaluation.loc[k, s, weight, method, 'Classifier'][
                            'score time SD'] = 0

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline,
                                random_state=1)

                            tic = process_time()
                            dummy_clf.fit(X_train, y_train)
                            fit_time = process_time()-tic

                            tic = process_time()
                            y_pred = dummy_clf.predict(X_test)
                            dummy_performance = (
                                functions.check_performance_holdout(y_test,
                                    y_pred, filename=None))
                            score_time = process_time()-tic

                            # Write fit and score times
                            classifier = 'Baseline '+baseline
                            evaluation.loc[k, s, weight, method, classifier][
                                'fit time MEAN'] = fit_time
                            evaluation.loc[k, s, weight, method, classifier][
                                'fit time SD'] = 0
                            evaluation.loc[k, s, weight, method, classifier][
                                'score time MEAN'] = score_time
                            evaluation.loc[k, s, weight, method, classifier][
                                'score time SD'] = 0

                            # Write performance of dummy 
                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan                                
                                evaluation.loc[k, s, weight, method,
                                    classifier][key] = (
                                        dummy_performance.loc[key]['value'])

                    elif method == 'cross-validation':
                        performance = check_performance_CV(knn, x, y, 5,
                            -1, os.path.join(path,
                                    '_'.join([filename, str(k), weight, s,
                                        method])))  # n_jobs=-1 ... use all CPUs

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline,
                                random_state=1)
                            dummy_performance = check_performance_CV(dummy_clf,
                                x, y, 5, -1, filename=None)  # n_jobs=-1 ... use all CPUs
  
                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan   
                                classifier = 'Baseline '+baseline
                                evaluation.loc[k, s, weight, method,
                                    classifier][key] = (
                                        dummy_performance.loc[key]['value'])

                    # For both methods identical
                    for key in performance.index:
                        if key not in evaluation:
                            evaluation[key] = np.nan
                        evaluation.loc[k, s, weight, method, 'Classifier'][
                            key] = performance.loc[key]['value']

    print(evaluation)
    evaluation.transpose().to_csv(os.path.join(path,
        filename + '_evaluation.csv'), sep=';', decimal=',')
    evaluation.to_hdf(os.path.join(path,
        filename + '_evaluation.h5'), key='evaluation', mode='w')
    
    return

def plot_evaluation_knn(path: str = None, filename: str = None):

    evaluation = pd.read_hdf(os.path.join(path,
        filename + '_evaluation.h5'), key='evaluation')

    list_k = evaluation.index.levels[0].to_list()
    scalings = evaluation.index.levels[1].to_list()
    weights = evaluation.index.levels[2].to_list()
    validation_methods = evaluation.index.levels[3].to_list()
    classifiers = evaluation.index.levels[4].to_list()

    # Plot evaluation parameters over parameters of algorithm

    for classifier in classifiers:
        for s in scalings:
            with sns.color_palette(n_colors=len(weights)*len(validation_methods)):
                fig, ax = plt.subplots(int(len(evaluation.keys())/2), 1,
                    sharex='all', tight_layout=True, figsize=(8,12))

            linestyle_cycle = ['-', '--', '-.', ':'] * 3  # to have enough elements (quick&dirty)
            marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*'] * 3  # to have enough elements (quick&dirty)

            lines = []
            for pos in range(int(len(evaluation.keys())/2)):
                ax[pos].set_title(evaluation.keys()[2*pos].replace(' MEAN', '').
                    replace('_', ' '))
                ax[pos].grid(True)

                lines = []
                for i, weight in enumerate(weights):
                    for j, method in enumerate(validation_methods):
                        MEAN = evaluation.loc[(slice(None), s,
                                        weight, method, classifier),
                                        evaluation.keys()[2*pos]].to_numpy()
                        SD = evaluation.loc[(slice(None), s,
                                        weight, method, classifier),
                                        evaluation.keys()[2*pos+1]].to_numpy()

                        lines.append(ax[pos].plot(list_k,
                            MEAN,
                            marker=marker_cycle[i+j],
                            linestyle=linestyle_cycle[i+j],
                            label=', '.join([method, weight]))[0])
                        ax[pos].fill_between(list_k, MEAN-SD, MEAN+SD,
                            alpha=0.3)
                if pos > 1:
                    ax[pos].set_ylim(0, 1)

            plt.xlabel(r'$k$')

            plt.subplots_adjust(hspace=2.4)
            fig.legend(  # title='hidden layer sizes',
                handles=lines, ncol=2, bbox_to_anchor=(0.5, -0.005),
                bbox_transform=fig.transFigure, loc='upper center',
                borderaxespad=0.1)
        
            fig.savefig(os.path.join(path, filename + '_' +
                '_'.join(['evaluation', s, classifier]) + '.png'),
                format='png', dpi=200, bbox_inches='tight')
            plt.close(fig)

    return


def gnb(X: np.array, y: np.array, test_size, random_state,
        validation_methods=['holdout', 'cross-validation'],
        path: str = None, filename: str = None):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state)

    index = pd.MultiIndex.from_product([validation_methods],
                                       names=['validation method'])
    evaluation = pd.DataFrame(index=index)

    # Change parameters
    # For holdout
    xtrain = X_train
    xtest = X_test
    # For k-fold CV
    x = X

    for method in validation_methods:
        gnb = GaussianNB()
        if method == 'holdout':

            tic = process_time()
            gnb.fit(xtrain, y_train)
            fit_time = process_time()-tic

            tic = process_time()
            y_pred_knn = gnb.predict(xtest)
            performance = (
                check_performance_holdout(y_test, y_pred_knn,
                    os.path.join(path,
                        '_'.join([filename, method])))
                )
            score_time = process_time()-tic

            # Add empty entries for fit and score times (mean, SD)
            for entry1 in ['fit', 'score']:
                for entry2 in ['MEAN', 'SD']:
                    entry = ' '.join([entry1, 'time', entry2])
                    if entry not in evaluation:
                        evaluation[entry] = np.nan
            # Write fit and score times
            evaluation.loc[method][
                'fit time MEAN'] = fit_time
            evaluation.loc[method][
                'fit time SD'] = 0
            evaluation.loc[method][
                'score time MEAN'] = score_time
            evaluation.loc[method][
                'score time SD'] = 0

            # Significance test against baselines
            strategies = ['stratified', 'uniform']
            for strategy in strategies:
                dummy_clf = DummyClassifier(strategy=strategy,
                    random_state=1)

                tic = process_time()
                dummy_clf.fit(X_train, y_train)
                fit_time = process_time()-tic

                tic = process_time()
                y_pred = dummy_clf.predict(X_test)
                dummy_performance = (
                    functions.check_performance_holdout(y_test,
                        y_pred, filename=None))
                score_time = process_time()-tic
                # Add empty entries for fit and score times (mean, SD)
                for entry1 in ['fit', 'score']:
                    for entry2 in ['MEAN', 'SD']:
                        entry = ' '.join(['dummy', strategy,
                            entry1, 'time', entry2])
                        if entry not in evaluation:
                            evaluation[entry] = np.nan
                # Write fit and score times
                prefix = 'dummy '+strategy+' '
                evaluation.loc[method][
                    prefix+'fit time MEAN'] = fit_time
                evaluation.loc[method][
                    prefix+'fit time SD'] = 0
                evaluation.loc[method][
                    prefix+'score time MEAN'] = score_time
                evaluation.loc[method][
                    prefix+'score time SD'] = 0

                # print(dummy_performance)    
                for key in dummy_performance.index:
                    label = ' '.join(['dummy', strategy, key])
                    if label not in evaluation:
                        evaluation[label] = np.nan
                    evaluation.loc[method][
                        label] = dummy_performance.loc[key]['value']

        elif method == 'cross-validation':
            performance = check_performance_CV(gnb, x, y, 5,
                -1, os.path.join(path,
                        '_'.join([filename, method])))  # n_jobs=-1 ... use all CPUs

            # Significance test against baselines
            strategies = ['stratified', 'uniform']
            for strategy in strategies:
                dummy_clf = DummyClassifier(strategy=strategy,
                    random_state=1)
                dummy_performance = check_performance_CV(dummy_clf,
                    x, y, 5, -1, filename=None)  # n_jobs=-1 ... use all CPUs

                for key in dummy_performance.index:
                    label = ' '.join(['dummy', strategy, key])
                    if label not in evaluation:
                        evaluation[label] = np.nan
                    evaluation.loc[method][
                        label] = dummy_performance.loc[key]['value']

        print(evaluation)
        # For both methods identical
        for key in performance.index:
            if key not in evaluation:
                evaluation[key] = np.nan

            print(evaluation.loc[method][key])
            print(performance.loc[key]['value'])

            evaluation.loc[method][
                key] = performance.loc[key]['value']

            print(evaluation.loc[method][key])
        stophere
        
    print(evaluation)
    evaluation.transpose().to_csv(os.path.join(path,
        filename + '_evaluation.csv'), sep=';', decimal=',')

    # # Plot evaluation parameters over parameters of algorithm
    # with sns.color_palette(n_colors=len(evaluation.keys())):
    #     fig, ax = plt.subplots(len(evaluation.keys()), 1, sharex='all', tight_layout=True, figsize=(8,19))

    # for pos, key in enumerate(evaluation.keys()):
    #     # ax = fig.add_subplot(5, 1, pos + 1)
    #     ax[pos].set_title(key)
    #     ax[pos].grid(True)

    #     ax[pos].plot(evaluation.loc[key].to_numpy(),
    #                  marker='o', linestyle='-', label='scaled, unif')

    # plt.xlabel(r'$k$')
    # plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.4))
    # fig.savefig(os.path.join(path, filename + '_evaluation.png'), format='png', dpi=300)
    # plt.close(fig)

    return


# def decision_tree(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
#                   list_max_depth=[1, 10, 30, 50, 100, 300], list_min_weight_fraction_leaf=[.0, .125, .25, .375, .5],
#                   list_min_samples_leaf=[1, 10, 100, 200], path: str = None, filename: str = None):
#     index = pd.MultiIndex.from_product([list_max_depth,
#                                         list_min_samples_leaf,
#                                         list_min_weight_fraction_leaf],
#                                        names=['max_depth', 'min_samples_leaf', 'min_weight_fraction_leaf'])
#     dt_errors = pd.DataFrame(index=index, columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])

#     for max_depth in list_max_depth:
#         for min_samples_leaf in list_min_samples_leaf:
#             for min_weight_fraction_leaf in list_min_weight_fraction_leaf:
#                 xtrain = X_train
#                 xtest = X_test

#                 dt = tree.DecisionTreeRegressor(
#                     max_depth=max_depth,
#                     min_samples_leaf=min_samples_leaf,
#                     min_weight_fraction_leaf=min_weight_fraction_leaf)
#                 dt.fit(xtrain, Y_train)
#                 y_pred_dt = dt.predict(xtest)

#                 errors = functions.check_performance_holdout(Y_test, y_pred_dt,
#                                                      os.path.join(path,
#                                                                   filename + '_' + str(max_depth) + '_' + str(
#                                                                       min_samples_leaf) + '_' + str(
#                                                                       min_weight_fraction_leaf)))
#                 dt_errors.loc[max_depth,
#                               min_samples_leaf,
#                               min_weight_fraction_leaf][:] = errors
#                 del xtrain, xtest

#     print(dt_errors)
#     dt_errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

#     # Plot errors over parameters of algorithm
#     for key3 in list_min_weight_fraction_leaf:
#         with sns.color_palette(n_colors=len(dt_errors.keys())):
#             fig, ax = plt.subplots(len(dt_errors.keys()), 1, sharex=True, tight_layout=True)
#             # ax = fig.add_subplot()
#         linestyle_cycle = ['-', '--', '-.', ':'] * 3  # to have enough elements (quick&dirty)
#         marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*'] * 3  # to have enough elements (quick&dirty)
#         for idx, key2 in enumerate(list_min_samples_leaf):
#             linestyle = linestyle_cycle[idx]
#             marker = marker_cycle[idx]
#             for pos, key in enumerate(dt_errors.keys()):
#                 # ax = fig.add_subplot(5, 1, pos + 1)
#                 ax[pos].set_title(key)
#                 ax[pos].grid(True)
#                 ax[pos].plot(list_max_depth, dt_errors.loc[(slice(None), key2, key3), key].to_numpy(),
#                              marker=marker, linestyle=linestyle, label=str(key2))

#         plt.subplots_adjust(hspace=2.4)
#         plt.xlabel('max_depth')
#         plt.legend(title='min_weight_fraction_leaf: ' + str(key3) + '\nmin_samples_leaf:',
#                    ncol=len(list_min_samples_leaf), loc='upper center', bbox_to_anchor=(0.5, -0.7))
#         fig.savefig(os.path.join(path, filename + '_errors_' + str(key3) + '.png'),
#                     format='png', dpi=200, bbox_inches='tight')
#         plt.close(fig)


# def mlp(X_train: np.array, X_test: np.array, Y_train: np.array, Y_test: np.array,
#         max_iter: int = 800, solver: str = 'lbfgs', list_alpha=[1e-7, 1e-4, 1e-1], list_hidden_layer_sizes=[[10]],
#         path: str = None, filename: str = None):  # list_neurons_per_hidden_layer, list_no_of_hidden_layers,

#     index = pd.MultiIndex.from_product(
#         [list_alpha, [str(x) for x in list_hidden_layer_sizes]],
#         names=['alpha', 'hidden_layer_sizes'])
#     mlp_errors = pd.DataFrame(index=index,
#                               columns=['MAE', 'MAPE', 'MSE', 'RMSE', 'EV'])
#     # Change parameters
#     for alpha in list_alpha:
#         for hidden_layer_sizes in list_hidden_layer_sizes:
#             mlp = neural_network.MLPRegressor(solver=solver,
#                                               max_iter=max_iter,
#                                               alpha=alpha,
#                                               hidden_layer_sizes=hidden_layer_sizes,
#                                               verbose=True,
#                                               random_state=5)

#             mlp.fit(X_train, Y_train)
#             y_pred_mlp = mlp.predict(X_test)

#             errors = functions.check_performance_holdout(Y_test, y_pred_mlp,
#                                                  os.path.join(path, filename + '_' + str(alpha) + '_' + str(
#                                                      hidden_layer_sizes)))
#             mlp_errors.loc[alpha, str(hidden_layer_sizes)][:] = errors

#     print(mlp_errors)
#     mlp_errors.transpose().to_csv(os.path.join(path, filename + '_errors.csv'), sep=';', decimal=',')

#     # Plot errors over parameters of algorithm
#     with sns.color_palette(n_colors=len(mlp_errors.keys())):
#         fig, ax = plt.subplots(len(mlp_errors.keys()), 1, sharex=True, tight_layout=True)

#     linestyle_cycle = ['-', '--', '-.', ':'] * 3  # to have enough elements (quick&dirty)
#     marker_cycle = ['o', 'o', 'o', 'o', '*', '*', '*', '*'] * 3  # to have enough elements (quick&dirty)

#     lines = []
#     for pos, key in enumerate(mlp_errors.keys()):
#         ax[pos].set_title(key)
#         ax[pos].grid(True)
#         lines = []
#         for idx, key2 in enumerate(list_hidden_layer_sizes):
#             lines.append(ax[pos].semilogx(list_alpha, mlp_errors.loc[(slice(None), str(key2)), key].to_numpy(),
#                                           marker=marker_cycle[idx], linestyle=linestyle_cycle[idx],
#                                           label=str(key2))[0])

#     # plt.ylim([0, 1])
#     plt.xlabel(r'$\alpha$')
#     fig.legend(title='hidden layer sizes', handles=lines, ncol=len(list_hidden_layer_sizes),
#                bbox_to_anchor=(0.5, -0.06), bbox_transform=fig.transFigure, loc='lower center', borderaxespad=0.1)
#     fig.savefig(os.path.join(path, filename + '_errors.png'), format='png', dpi=200, bbox_inches='tight')
#     plt.close(fig)
