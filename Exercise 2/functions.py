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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix   # not available in 0.21.3
from helper import plot_confusion_matrix  # source copied from newest sklearn
from sklearn.metrics import classification_report

from sklearn.dummy import DummyClassifier

import functions
import os
import seaborn as sns

from time import process_time


def check_performance_holdout(classifier, X_test: np.array, y_test: np.array, y_pred: np.array, filename=None):
    # Plot confusion matrix
    plot_confusion_matrix(classifier, X_test, y_test, normalize=None)

    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + "_confusion_matrix.png", format='png')
        plt.close()

        # Evaluation metrics
    dict_report = classification_report(y_test, y_pred, output_dict=True)

    # Generate columns for result
    scores = ['accuracy', 'precision', 'recall', 'f1-score']
    averages = ['', 'macro avg', 'macro avg', 'macro avg']
    index_elements = list(zip(scores, averages))

    scoring = [' '.join([score, average]).strip().
                   replace(' avg', '').replace(' ', '_').replace('-', '_').
                   replace('f1_score', 'f1')
               for score, average in index_elements]

    index = []
    # Add mean and standard deviations to index
    for score in scoring:
        index.append(score + ' MEAN')
        index.append(score + ' SD')
    # # Add mean values and standard deviation of fit and score times to index 
    # for time in ['fit', 'score']:
    #     for stat in ['MEAN', 'SD']:
    #         index.append(time+' time '+stat)
    result = pd.DataFrame(index=index, columns=['value'])

    # Pick results from report
    for idx, (score, average) in enumerate(index_elements):
        # Mean value 
        if score == 'accuracy':
            result.loc[index[idx * 2]]['value'] = dict_report[score]
        else:
            result.loc[index[idx * 2]]['value'] = dict_report[average][score]
        # Standard deviation
        result.loc[index[idx * 2 + 1]]['value'] = 0

    print(result)
    print()

    if filename is not None:
        result.to_csv(filename + '.txt', float_format='%.2f', sep='\t', header=False)

    return result


def check_performance_CV(classifier, X: np.array, y: np.array, n_splits=5, n_jobs=-1, filename=None):
   
    scores = ['accuracy', 'precision', 'recall', 'f1-score']
    averages = ['', 'macro avg', 'macro avg', 'macro avg']
    index_elements = list(zip(scores, averages))

    scoring = [' '.join([score, average]).strip().
                   replace(' avg', '').replace(' ', '_').replace('-', '_').
                   replace('f1_score', 'f1')
               for score, average in index_elements]

    index = []
    # Add mean and standard deviations to index
    for score in scoring:
        index.append(score + ' MEAN')
        index.append(score + ' SD')
    # Add mean values and standard deviation of fit and score times to index 
    for time in ['fit', 'score']:
        for stat in ['MEAN', 'SD']:
            index.append(time + ' time ' + stat)
    result = pd.DataFrame(index=index, columns=['value'])

    # Provide train/test indices to split data in train/test sets
    skf = StratifiedKFold(n_splits=n_splits)
    # skf.get_n_splits(X, y)

    dict_CV_results = cross_validate(classifier, X, y, cv=skf, scoring=scoring, n_jobs=n_jobs, return_estimator=True)

    # Plot confusion matrix for estimator with highest accuracy
    idx_max = np.argmax(dict_CV_results['test_accuracy'])

    best_estimator = dict_CV_results['estimator'][idx_max]
    # Get corresponding test_set
    test_sets = [set for set in skf.split(X, y)]
    best_set = test_sets[idx_max][1]
    # Plot
    plot_confusion_matrix(best_estimator, X[best_set], y[best_set], normalize=None)

    if filename is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename + "_confusion_matrix.png", format='png')
        plt.close()

    # Pick results from cross_validate
    for idx in range(len(index_elements)):
        # Mean value 
        result.loc[index[idx * 2]]['value'] = np.mean(dict_CV_results['test_' + scoring[idx]])
        # Standard deviation
        result.loc[index[idx * 2 + 1]]['value'] = np.std(dict_CV_results['test_' + scoring[idx]])

    # Pick fit_time and score_time from cross_validate
    for time in ['fit', 'score']:
        # Mean value 
        result.loc[time + ' time MEAN']['value'] = np.mean(dict_CV_results[time + '_time'])
        # Standard deviation
        result.loc[time + ' time SD']['value'] = np.std(dict_CV_results[time + '_time'])

    print(result)
    print()

    if filename is not None:
        result.to_csv(filename + '.txt', float_format='%.2f', sep='\t', header=False)

    return result


def knn(X: np.array, y: np.array, test_size=0.2, random_state=1,
        list_k=[1, 3, 5], scaling: bool = True,
        weights=['uniform', 'distance'],
        validation_methods=['holdout', 'cross-validation'],
        baselines=['stratified', 'uniform'],
        path: str = None, filename: str = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
        validation_methods, ['Classifier'] + ['Baseline ' + i for i in baselines]],
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
                        fit_time = process_time() - tic

                        tic = process_time()
                        y_pred_knn = knn.predict(xtest)
                        performance = (check_performance_holdout(knn, xtest, y_test, y_pred_knn,
                                                                 os.path.join(path, '_'.join(
                                                                     [filename, str(k), weight, s, method]))))
                        score_time = process_time() - tic

                        # Write fit and score times
                        evaluation.loc[k, s, weight, method, 'Classifier']['fit time MEAN'] = fit_time
                        evaluation.loc[k, s, weight, method, 'Classifier']['fit time SD'] = 0
                        evaluation.loc[k, s, weight, method, 'Classifier']['score time MEAN'] = score_time
                        evaluation.loc[k, s, weight, method, 'Classifier']['score time SD'] = 0

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline, random_state=random_state)

                            tic = process_time()
                            dummy_clf.fit(X_train, y_train)
                            fit_time = process_time() - tic

                            tic = process_time()
                            y_pred = dummy_clf.predict(X_test)
                            dummy_performance = (check_performance_holdout(dummy_clf, xtest, y_test, y_pred,
                                                                           os.path.join(path, '_'.join(
                                                                               [filename, str(k), weight, s, method,
                                                                                baseline]))))
                            score_time = process_time() - tic

                            # Write fit and score times
                            classifier = 'Baseline ' + baseline
                            evaluation.loc[k, s, weight, method, classifier]['fit time MEAN'] = fit_time
                            evaluation.loc[k, s, weight, method, classifier]['fit time SD'] = 0
                            evaluation.loc[k, s, weight, method, classifier]['score time MEAN'] = score_time
                            evaluation.loc[k, s, weight, method, classifier]['score time SD'] = 0

                            # Write performance of dummy 
                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan
                                evaluation.loc[k, s, weight, method, classifier][key] = (
                                    dummy_performance.loc[key]['value'])

                    elif method == 'cross-validation':
                        performance = check_performance_CV(knn, x, y, 5, -1,
                            os.path.join(path, '_'.join([filename, str(k),
                                weight, s, method])))  # n_jobs=-1 ... use all CPUs

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline, random_state=random_state)
                            dummy_performance = check_performance_CV(dummy_clf,
                                x, y, 5, -1, os.path.join(path,
                                '_'.join( [filename, str(k), weight, s, method,
                                    baseline])))  # n_jobs=-1 ... use all CPUs

                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan
                                classifier = 'Baseline ' + baseline
                                evaluation.loc[k, s, weight, method,
                                    classifier][key] = (
                                        dummy_performance.loc[key]['value'])

                    # Write performance of classifier 
                    for key in performance.index:
                        if key not in evaluation:
                            evaluation[key] = np.nan
                        evaluation.loc[k, s, weight, method,
                            'Classifier'][key] = performance.loc[key]['value']

    print(evaluation)
    evaluation.transpose().to_csv(os.path.join(path, filename + '_evaluation.csv'), sep=';', decimal=',')
    evaluation.to_hdf(os.path.join(path, filename + '_evaluation.h5'), key='evaluation', mode='w')

    return


def plot_evaluation_knn(path: str = None, filename: str = None):

    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')
    # Read MultiIndex for the loops
    list_k = evaluation.index.levels[0].to_list()
    scalings = evaluation.index.levels[1].to_list()
    weights = evaluation.index.levels[2].to_list()
    validation_methods = evaluation.index.levels[3].to_list()
    classifiers = evaluation.index.levels[4].to_list()
    classifiers.sort(reverse=True)

    # Divide DataFrame into efficiency (times) and effectiveness (scores)
    # for separate figures
    time_keys = ['fit time MEAN', 'fit time SD', 'score time MEAN', 'score time SD']
    score_keys = evaluation.columns.to_list()
    for x in time_keys:
        score_keys.remove(x)
    performances = {'efficiency': evaluation[time_keys].copy(),
        'effectiveness': evaluation[score_keys].copy()}
    del evaluation

    for key, evaluation in performances.items():
        # Plot evaluation parameters over parameters of algorithm
          for s in scalings:
            with sns.color_palette(n_colors=len(weights) * len(validation_methods)):
                fig, ax = plt.subplots(int(len(evaluation.keys()) / 2), 1,
                    sharex='all', tight_layout=True, figsize=(7, 8))

            linestyle_cycle = ['-', '--'] * 3
            marker_cycle = ['o', '+', 'x']

            lines = []
            for pos in range(int(len(evaluation.keys()) / 2)):
                ax[pos].set_title(evaluation.keys()[2 * pos].replace(' MEAN', '').
                                replace('_', ' '))
                ax[pos].grid(True)

                lines = []
                # Plot classifier and baselines on the same axes
                for idx, classifier in enumerate(classifiers):
                    for i, weight in enumerate(weights):
                        for j, method in enumerate(validation_methods):
                            MEAN = evaluation.loc[(slice(None), s, weight,
                                method, classifier),
                                evaluation.keys()[2 * pos]].to_numpy()
                            SD = evaluation.loc[(slice(None), s, weight,
                                method, classifier),
                                evaluation.keys()[2 * pos + 1]].to_numpy()

                            lines.append(ax[pos].plot(list_k, MEAN,
                                marker=marker_cycle[idx],
                                linestyle=linestyle_cycle[j],
                                label=', '.join([classifier, weight, method]).
                                    replace('Classifier', 'Classif.').
                                    replace('Baseline', 'Basel.').
                                    replace('distance', 'dist').
                                    replace('uniform', 'unif').
                                    replace('stratified', 'strat.').
                                    replace('cross-validation', 'CV'))[
                                    0])
                            ax[pos].fill_between(list_k, MEAN - SD, MEAN + SD,
                                alpha=0.3)
                if key == 'efficiency':
                    # Upper limit shall be maximum of times (mean + sd) to
                    # be the same in all figures
                    ax[pos].set_ylim(0, (evaluation[evaluation.keys()[2 * pos]]+
                            evaluation[evaluation.keys()[2 * pos + 1]]).max())
                elif key == 'effectiveness':
                    # Limits shall be range of scores (mean +- sd) to
                    # be the same in all figures
                    ymin = max(0, (evaluation[evaluation.keys()[2 * pos]] - 
                            evaluation[evaluation.keys()[2 * pos + 1]]).min())
                    ymax = min(1, (evaluation[evaluation.keys()[2 * pos]] + 
                            evaluation[evaluation.keys()[2 * pos + 1]]).max())
                    ax[pos].set_ylim(ymin, ymax)

            plt.xlabel(r'$k$')

            plt.subplots_adjust(hspace=2.4)
            fig.legend(
                handles=lines, ncol=3, bbox_to_anchor=(0.5, -0.005),
                bbox_transform=fig.transFigure, loc='upper center',
                borderaxespad=0.1)

            fig.savefig(os.path.join(path, filename + '_' +
                '_'.join(['evaluation', s, key]) + '.png'), format='png',
                dpi=200, bbox_inches='tight')
            plt.close(fig)

    return


def plot_accuracy_knn(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    scalings = evaluation.index.levels[1].to_list()
    weights = evaluation.index.levels[2].to_list()

    fig = plt.figure()
    marker_cycle = ['o', '+', 'x', '1']
    idx = 0
    for scaling in scalings:
        for weight in weights:
            label = ', '.join([scaling, weight])
            rows = (slice(None), scaling, weight, 'cross-validation', 'Classifier')
            plt.scatter(
                evaluation.loc[rows, ('accuracy MEAN')],
                evaluation.loc[rows, ('accuracy SD')],
                label=label,
                marker=marker_cycle[idx])
            idx = idx + 1
    plt.ylim(bottom=0)
    plt.title('Accuracy from cross-validation')
    plt.xlabel('mean value')
    plt.ylabel('standard deviation')
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_accuracy.png'), format='png', dpi=200,
        bbox_inches='tight')
    return


def plot_efficiency_knn(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    scalings = evaluation.index.levels[1].to_list()
    weights = evaluation.index.levels[2].to_list()

    fig = plt.figure()
    marker_cycle = ['o', '+', 'x', '1']
    idx = 0
    for scaling in scalings:
        for weight in weights:
            label = ', '.join([scaling, weight])
            rows = (slice(None), scaling, weight, 'cross-validation', 'Classifier')
            plt.errorbar(
                x=evaluation.loc[rows, ('fit time MEAN')],
                y=evaluation.loc[rows, ('score time MEAN')],
                xerr=evaluation.loc[rows, ('fit time SD')],
                yerr=evaluation.loc[rows, ('score time SD')],
                label=label,
                marker=marker_cycle[idx],
                linestyle='') 
            idx = idx + 1
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Efficiency from cross-validation')
    plt.xlabel('fit time')
    plt.ylabel('score time')
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_efficiency.png'),
        format='png', dpi=200, bbox_inches='tight')
    return


def gnb(X: np.array, y: np.array, test_size=0.2, random_state=1,
        validation_methods=['holdout', 'cross-validation'],
        baselines=['stratified', 'uniform'],
        path: str = None, filename: str = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state)

    index = pd.MultiIndex.from_product([validation_methods,
        ['Classifier'] + ['Baseline ' + i for i in baselines]],
        names=['validation method', 'classifier'])
    evaluation = pd.DataFrame(index=index)
    # Add empty entries for fit and score times (mean, SD)
    for entry1 in ['fit', 'score']:
        for entry2 in ['MEAN', 'SD']:
            entry = ' '.join([entry1, 'time', entry2])
            if entry not in evaluation:
                evaluation[entry] = np.nan

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
            fit_time = process_time() - tic

            tic = process_time()
            y_pred_knn = gnb.predict(xtest)
            performance = (
                check_performance_holdout(gnb, xtest, y_test, y_pred_knn,
                                          os.path.join(path,
                                                       '_'.join([filename, method])))
            )
            score_time = process_time() - tic

            # Write fit and score times
            evaluation.loc[method, 'Classifier']['fit time MEAN'] = fit_time
            evaluation.loc[method, 'Classifier']['fit time SD'] = 0
            evaluation.loc[method, 'Classifier']['score time MEAN'] = score_time
            evaluation.loc[method, 'Classifier']['score time SD'] = 0

            # Significance test against baselines
            for baseline in baselines:
                dummy_clf = DummyClassifier(strategy=baseline, random_state=random_state)

                tic = process_time()
                dummy_clf.fit(X_train, y_train)
                fit_time = process_time() - tic

                tic = process_time()
                y_pred = dummy_clf.predict(X_test)
                dummy_performance = (
                    check_performance_holdout(dummy_clf, xtest, y_test, y_pred,
                        os.path.join(path, '_'.join([filename, method,
                        baseline]))))
                score_time = process_time() - tic

                # Write fit and score times
                classifier = 'Baseline ' + baseline
                evaluation.loc[method, classifier]['fit time MEAN'] = fit_time
                evaluation.loc[method, classifier]['fit time SD'] = 0
                evaluation.loc[method, classifier]['score time MEAN'] = score_time
                evaluation.loc[method, classifier]['score time SD'] = 0

                # Write performance of dummy    
                for key in dummy_performance.index:
                    if key not in evaluation:
                        evaluation[key] = np.nan
                    evaluation.loc[method, classifier][key] = (
                        dummy_performance.loc[key]['value'])

        elif method == 'cross-validation':
            performance = check_performance_CV(gnb, x, y, 5, -1,
                os.path.join(path, '_'.join([filename, method])))  # n_jobs=-1 ... use all CPUs

            # Significance test against baselines
            for baseline in baselines:
                dummy_clf = DummyClassifier(strategy=baseline,
                    random_state=random_state)
                dummy_performance = check_performance_CV(dummy_clf, x, y, 5,
                    -1, os.path.join(path, '_'.join([filename, method,
                        baseline])))  # n_jobs=-1 ... use all CPUs

                for key in dummy_performance.index:
                    if key not in evaluation:
                        evaluation[key] = np.nan
                    classifier = 'Baseline ' + baseline
                    evaluation.loc[method, classifier][key] = (
                        dummy_performance.loc[key]['value'])

        # Write performance of classifier
        for key in performance.index:
            if key not in evaluation:
                evaluation[key] = np.nan
            evaluation.loc[method, 'Classifier'][key] = (
                performance.loc[key]['value'])

    print(evaluation)
    evaluation.transpose().to_csv(os.path.join(path, filename + '_evaluation.csv'), sep=';', decimal=',')
    evaluation.to_hdf(os.path.join(path, filename + '_evaluation.h5'), key='evaluation', mode='w')

    return


def plot_evaluation_gnb(path: str = None, filename: str = None):

    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')
    # Read MultiIndex for the loops
    validation_methods = evaluation.index.levels[0].to_list()
    classifiers = evaluation.index.levels[1].to_list()
    classifiers.sort(reverse=True)

    # Divide DataFrame into efficiency (times) and effectiveness (scores)
    # for separate figures
    time_keys = ['fit time MEAN', 'fit time SD', 'score time MEAN', 'score time SD']
    score_keys = evaluation.columns.to_list()
    for x in time_keys:
        score_keys.remove(x)
    performances = {'efficiency': evaluation[time_keys].copy(),
        'effectiveness': evaluation[score_keys].copy()}
    del evaluation

    for key, evaluation in performances.items():
        # Plot evaluation parameters over parameters of algorithm
        with sns.color_palette(n_colors=len(validation_methods)):
            fig, ax = plt.subplots(int(len(evaluation.keys()) / 2), 1,
                sharex='all', tight_layout=True, figsize=(7, 8))

        linestyle_cycle = ['-', '--'] * 3
        marker_cycle = ['o', '+', 'x']

        lines = []
        for pos in range(int(len(evaluation.keys()) / 2)):
            ax[pos].set_title(evaluation.keys()[2 * pos].replace(' MEAN', '').
                            replace('_', ' '))
            ax[pos].grid(True)

            lines = []
            # Plot classifier and baselines on the same axes
            for idx, classifier in enumerate(classifiers):
                for j, method in enumerate(validation_methods):
                    MEAN = evaluation.loc[(method, classifier),
                        evaluation.keys()[2 * pos]]
                    SD = evaluation.loc[(method, classifier),
                        evaluation.keys()[2 * pos + 1]]

                    lines.append(ax[pos].plot(np.array([0, 1, 2]),
                        np.array([MEAN, MEAN, MEAN]),
                        marker=marker_cycle[idx],
                        linestyle=linestyle_cycle[j],
                        label=', '.join([classifier, method]).
                            replace('Classifier', 'Classif.').
                            replace('Baseline', 'Basel.').
                            replace('uniform', 'unif').
                            replace('stratified', 'strat.').
                            replace('cross-validation', 'CV'))[
                            0])
                    ax[pos].fill_between(np.array([0, 1, 2]),
                        np.array([MEAN - SD, MEAN - SD, MEAN - SD]),
                        np.array([MEAN + SD, MEAN + SD, MEAN + SD]),
                        alpha=0.3)
            if key == 'efficiency':
                # Upper limit shall be maximum of times (mean + sd) to
                # be the same in all figures
                ax[pos].set_ylim(0, (evaluation[evaluation.keys()[2 * pos]]+
                        evaluation[evaluation.keys()[2 * pos + 1]]).max())
            elif key == 'effectiveness':
                # Limits shall be range of scores (mean +- sd) to
                # be the same in all figures
                ymin = max(0, (evaluation[evaluation.keys()[2 * pos]] - 
                        evaluation[evaluation.keys()[2 * pos + 1]]).min())
                ymax = min(1, (evaluation[evaluation.keys()[2 * pos]] + 
                        evaluation[evaluation.keys()[2 * pos + 1]]).max())
                ax[pos].set_ylim(ymin, ymax)
            plt.xlim(0.5, 1.5)
            plt.xticks(ticks=[1], labels=' ')
        plt.subplots_adjust(hspace=2.4)
        fig.legend(
            handles=lines, ncol=3, bbox_to_anchor=(0.5, -0.005),
            bbox_transform=fig.transFigure, loc='upper center',
            borderaxespad=0.1)

        fig.savefig(os.path.join(path, filename + '_' + 
            '_'.join(['evaluation', key]) + '.png'), format='png', dpi=200,
            bbox_inches='tight')
        plt.close(fig)

    return


def plot_accuracy_gnb(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    fig = plt.figure()
    rows = ('cross-validation', 'Classifier')
    plt.scatter(
        evaluation.loc[rows, ('accuracy MEAN')],
        evaluation.loc[rows, ('accuracy SD')],
        marker='o')
    plt.ylim(bottom=0)
    plt.title('Accuracy from cross-validation')
    plt.xlabel('mean value')
    plt.ylabel('standard deviation')
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_accuracy.png'),
        format='png', dpi=200, bbox_inches='tight')
    return


def plot_efficiency_gnb(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    fig = plt.figure()
    rows = ('cross-validation', 'Classifier')
    plt.errorbar(
        x=evaluation.loc[rows, ('fit time MEAN')],
        y=evaluation.loc[rows, ('score time MEAN')],
        xerr=evaluation.loc[rows, ('fit time SD')],
        yerr=evaluation.loc[rows, ('score time SD')],
        marker='o',
        linestyle='') 
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Efficiency from cross-validation')
    plt.xlabel('fit time')
    plt.ylabel('score time')
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_efficiency.png'),
        format='png', dpi=200, bbox_inches='tight')
    return


def dt(X: np.array, y: np.array, test_size=0.2, random_state=1,
        list_max_depth=[1, 10], list_min_samples_split=[2, 20],
        list_min_samples_leaf=[1, 10],
        validation_methods=['holdout', 'cross-validation'],
        baselines=['stratified', 'uniform'],
        path: str = None, filename: str = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=random_state)

    index = pd.MultiIndex.from_product([list_max_depth, list_min_samples_split,
        list_min_samples_leaf, validation_methods,
        ['Classifier'] + ['Baseline ' + i for i in baselines]],
        names=['max_depth', 'list_min_samples_split', 'list_min_samples_leaf',
            'validation method', 'classifier'])
    evaluation = pd.DataFrame(index=index)
    # Add empty entries for fit and score times (mean, SD)
    for entry1 in ['fit', 'score']:
        for entry2 in ['MEAN', 'SD']:
            entry = ' '.join([entry1, 'time', entry2])
            if entry not in evaluation:
                evaluation[entry] = np.nan

    # Change parameters
    for max_depth in list_max_depth:
        for min_samples_split in list_min_samples_split:
            for min_samples_leaf in list_min_samples_leaf:
                # For holdout
                xtrain = X_train
                xtest = X_test
                # For k-fold CV
                x = X

                for method in validation_methods:
                    dt = DecisionTreeClassifier(max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state)
                    if method == 'holdout':
                        tic = process_time()
                        dt.fit(xtrain, y_train)
                        fit_time = process_time() - tic

                        tic = process_time()
                        y_pred_knn = dt.predict(xtest)
                        performance = (check_performance_holdout(dt, xtest,
                            y_test, y_pred_knn,
                            os.path.join(path, '_'.join(
                                [filename, str(max_depth), str(min_samples_split),
                                    str(min_samples_leaf), method]))))
                        score_time = process_time() - tic

                        # Write fit and score times
                        evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, 'Classifier']['fit time MEAN'] = fit_time
                        evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, 'Classifier']['fit time SD'] = 0
                        evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, 'Classifier']['score time MEAN'] = score_time
                        evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, 'Classifier']['score time SD'] = 0

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline, random_state=random_state)

                            tic = process_time()
                            dummy_clf.fit(X_train, y_train)
                            fit_time = process_time() - tic

                            tic = process_time()
                            y_pred = dummy_clf.predict(X_test)
                            dummy_performance = (check_performance_holdout(
                                dummy_clf, xtest, y_test, y_pred,
                                    os.path.join(path, '_'.join(
                                        [filename, str(max_depth),
                                            str(min_samples_split),
                                            str(min_samples_leaf), method,
                                            baseline]))))
                            score_time = process_time() - tic

                            # Write fit and score times
                            classifier = 'Baseline ' + baseline
                            evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, classifier]['fit time MEAN'] = fit_time
                            evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, classifier]['fit time SD'] = 0
                            evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, classifier]['score time MEAN'] = score_time
                            evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method, classifier]['score time SD'] = 0

                            # Write performance of dummy 
                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan
                                evaluation.loc[max_depth, min_samples_split,
                                    min_samples_leaf, method,
                                    classifier][key] = (
                                        dummy_performance.loc[key]['value'])

                    elif method == 'cross-validation':
                        performance = check_performance_CV(dt, x, y, 5, -1,
                            os.path.join(path, '_'.join([filename,
                                str(max_depth), str(min_samples_split),
                                str(min_samples_leaf), method])))  # n_jobs=-1 ... use all CPUs

                        # Significance test against baselines
                        for baseline in baselines:
                            dummy_clf = DummyClassifier(strategy=baseline, random_state=random_state)
                            dummy_performance = check_performance_CV(dummy_clf,
                                x, y, 5, -1, os.path.join(path,
                                '_'.join( [filename, str(max_depth),
                                    str(min_samples_split),
                                    str(min_samples_leaf), method, baseline])))  # n_jobs=-1 ... use all CPUs

                            for key in dummy_performance.index:
                                if key not in evaluation:
                                    evaluation[key] = np.nan
                                classifier = 'Baseline ' + baseline
                                evaluation.loc[max_depth, min_samples_split, min_samples_leaf, method,
                                    classifier][key] = (
                                        dummy_performance.loc[key]['value'])

                    # Write performance of classifier 
                    for key in performance.index:
                        if key not in evaluation:
                            evaluation[key] = np.nan
                        evaluation.loc[max_depth, min_samples_split,
                            min_samples_leaf, method, 'Classifier'][key] = (
                                performance.loc[key]['value'])

    print(evaluation)
    evaluation.transpose().to_csv(os.path.join(path, filename + '_evaluation.csv'), sep=';', decimal=',')
    evaluation.to_hdf(os.path.join(path, filename + '_evaluation.h5'), key='evaluation', mode='w')

    return


def plot_evaluation_dt(path: str = None, filename: str = None):

    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')
    # Read MultiIndex for the loops
    list_max_depth = evaluation.index.levels[0].to_list()
    list_min_samples_split = evaluation.index.levels[1].to_list()
    list_min_samples_leaf = evaluation.index.levels[2].to_list()
    validation_methods = evaluation.index.levels[3].to_list()
    classifiers = evaluation.index.levels[4].to_list()
    classifiers.sort(reverse=True)

    # Divide DataFrame into efficiency (times) and effectiveness (scores)
    # for separate figures
    time_keys = ['fit time MEAN', 'fit time SD', 'score time MEAN', 'score time SD']
    score_keys = evaluation.columns.to_list()
    for x in time_keys:
        score_keys.remove(x)
    performances = {'efficiency': evaluation[time_keys].copy(),
        'effectiveness': evaluation[score_keys].copy()}
    del evaluation

    for key, evaluation in performances.items():
        # Plot evaluation parameters over parameters of algorithm
        for min_samples_split in list_min_samples_split:
            for min_samples_leaf in list_min_samples_leaf:
                with sns.color_palette(n_colors=len(validation_methods)):
                        fig, ax = plt.subplots(int(len(evaluation.keys()) / 2),
                            1, sharex='all', tight_layout=True, figsize=(7, 8))

                linestyle_cycle = ['-', '--'] * 3
                marker_cycle = ['o', '+', 'x']

                lines = []
                for pos in range(int(len(evaluation.keys()) / 2)):
                    ax[pos].set_title(evaluation.keys()[2 * pos].
                        replace(' MEAN', '').replace('_', ' '))
                    ax[pos].grid(True)

                    lines = []
                    # Plot classifier and baselines on the same axes
                    for idx, classifier in enumerate(classifiers):
                        for j, method in enumerate(validation_methods):
                            MEAN = evaluation.loc[(slice(None),
                                min_samples_split, min_samples_leaf,
                                method, classifier),
                                evaluation.keys()[2 * pos]].to_numpy()
                            SD = evaluation.loc[(slice(None),
                                min_samples_split, min_samples_leaf,
                                method, classifier),
                                evaluation.keys()[2 * pos + 1]].to_numpy()

                            lines.append(ax[pos].plot(list_max_depth, MEAN,
                                marker=marker_cycle[idx],
                                linestyle=linestyle_cycle[j],
                                label=', '.join([classifier, method]).
                                    replace('Classifier', 'Classif.').
                                    replace('Baseline', 'Basel.').
                                    replace('uniform', 'unif').
                                    replace('stratified', 'strat.').
                                    replace('cross-validation', 'CV'))[
                                    0])
                            ax[pos].fill_between(list_max_depth, MEAN - SD,
                                MEAN + SD, alpha=0.3)
                    if key == 'efficiency':
                        # Upper limit shall be maximum of times (mean + sd) to
                        # be the same in all figures
                        ax[pos].set_ylim(0, (evaluation[evaluation.keys()[2 * pos]]+
                                evaluation[evaluation.keys()[2 * pos + 1]]).max())
                    elif key == 'effectiveness':
                        # Limits shall be range of scores (mean +- sd) to
                        # be the same in all figures
                        ymin = max(0, (evaluation[evaluation.keys()[2 * pos]] - 
                                evaluation[evaluation.keys()[2 * pos + 1]]).min())
                        ymax = min(1, (evaluation[evaluation.keys()[2 * pos]] + 
                                evaluation[evaluation.keys()[2 * pos + 1]]).max())
                        ax[pos].set_ylim(ymin, ymax)

                plt.xlabel('max_depth')

                plt.subplots_adjust(hspace=2.4)
                fig.legend(title='min_samples_split: ' + str(min_samples_split) +
                    '\nmin_samples_leaf: ' + str(min_samples_leaf),
                    handles=lines, ncol=3, bbox_to_anchor=(0.5, -0.005),
                    bbox_transform=fig.transFigure, loc='upper center',
                    borderaxespad=0.1)

                fig.savefig(os.path.join(path, filename + '_' +
                    '_'.join(['evaluation', str(min_samples_split),
                    str(min_samples_leaf), key]) + '.png'), format='png',
                    dpi=200, bbox_inches='tight')
                plt.close(fig)

    return


def plot_accuracy_dt(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    list_min_samples_split = evaluation.index.levels[1].to_list()
    list_min_samples_leaf = evaluation.index.levels[2].to_list()

    with sns.color_palette(n_colors=len(list_min_samples_leaf)):
        fig = plt.figure()
        marker_cycle = ['+', 'x', '1', '2']*4
        idx = 0
        for min_samples_split in list_min_samples_split:
            for min_samples_leaf in list_min_samples_leaf:
                label = ', '.join([str(min_samples_split), str(min_samples_leaf)])
                rows = (slice(None), min_samples_split, min_samples_leaf,
                    'cross-validation', 'Classifier')
                plt.scatter(
                    evaluation.loc[rows, ('accuracy MEAN')],
                    evaluation.loc[rows, ('accuracy SD')],
                    label=label,
                    marker=marker_cycle[idx])
                idx = idx + 1
    plt.ylim(bottom=0)
    plt.title('Accuracy from cross-validation')
    plt.xlabel('mean value')
    plt.ylabel('standard deviation')
    plt.legend(title='min_samples_split, min_samples_leaf:', ncol=4,
        bbox_to_anchor=(0.5, -0.005), bbox_transform=fig.transFigure,
        loc='upper center', borderaxespad=0.1)
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_accuracy.png'),
        format='png', dpi=200, bbox_inches='tight')
    return


def plot_efficiency_dt(path: str = None, filename: str = None):
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    list_min_samples_split = evaluation.index.levels[1].to_list()
    list_min_samples_leaf = evaluation.index.levels[2].to_list()

    with sns.color_palette(n_colors=len(list_min_samples_leaf)):
        fig = plt.figure()
        marker_cycle = ['+', 'x', '1', '2']*4
        idx = 0
        for min_samples_split in list_min_samples_split:
            for min_samples_leaf in list_min_samples_leaf:
                label = ', '.join([str(min_samples_split), str(min_samples_leaf)])
                rows = (slice(None), min_samples_split, min_samples_leaf,
                    'cross-validation', 'Classifier')
                plt.errorbar(
                    x=evaluation.loc[rows, ('fit time MEAN')],
                    y=evaluation.loc[rows, ('score time MEAN')],
                    xerr=evaluation.loc[rows, ('fit time SD')],
                    yerr=evaluation.loc[rows, ('score time SD')],
                    label=label,
                    marker=marker_cycle[idx],
                    linestyle='')            
                idx = idx + 1
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Efficiency from cross-validation')
    plt.xlabel('fit time')
    plt.ylabel('score time')
    plt.legend(title='min_samples_split, min_samples_leaf:', ncol=4,
        bbox_to_anchor=(0.5, -0.005), bbox_transform=fig.transFigure,
        loc='upper center', borderaxespad=0.1)
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, filename + '_efficiency.png'),
        format='png', dpi=200, bbox_inches='tight')
    return


def plot_accuracy(path: str=None, filenames: list=['knn', 'gnb', 'dt']):
    with sns.color_palette(n_colors=len(filenames)+2):
        fig = plt.figure()
        marker_cycle = ['+', 'x', '1', '2', '3']*3
        idx = 0
        # Plot Classifier
        for filename in filenames:
            evaluation = pd.read_hdf(os.path.join(path,  filename +
                '_evaluation.h5'), key='evaluation')

            rows = ()
            for name in evaluation.index.names:
                if name == 'validation method':
                    rows = rows + ('cross-validation', )
                elif name == 'classifier':
                    rows = rows + ('Classifier', )
                else:
                    rows = rows + (slice(None), )
            plt.scatter(
                evaluation.loc[rows, ('accuracy MEAN')],
                evaluation.loc[rows, ('accuracy SD')],
                label=filename,
                marker=marker_cycle[idx])

            idx = idx + 1

        # Plot baselines
        baselines = ['Baseline stratified', 'Baseline uniform']
        for baseline in baselines:
            rows = ()
            for name in evaluation.index.names:
                if name == 'validation method':
                    rows = rows + ('cross-validation', )
                elif name == 'classifier':
                    rows = rows + (baseline, )
                else:
                    rows = rows + (slice(None), )
            plt.scatter(
                evaluation.loc[rows, ('accuracy MEAN')],
                evaluation.loc[rows, ('accuracy SD')],
                label=baseline,
                marker=marker_cycle[idx])
            idx = idx + 1

    plt.ylim(bottom=0)
    plt.xlim(right=1)
    plt.title('Accuracy from cross-validation')
    plt.xlabel('mean value')
    plt.ylabel('standard deviation')
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, 'accuracy.png'), format='png', dpi=200,
        bbox_inches='tight')

    return

def plot_efficiency(path: str=None, filenames: list=['knn', 'gnb', 'dt']):
    with sns.color_palette(n_colors=len(filenames)+2):
        fig = plt.figure()
        marker_cycle = ['+', 'x', '1', '2', '3']*3
        idx = 0
        # Plot Classifier
        for filename in filenames:
            evaluation = pd.read_hdf(os.path.join(path,  filename +
                '_evaluation.h5'), key='evaluation')

            rows = ()
            for name in evaluation.index.names:
                if name == 'validation method':
                    rows = rows + ('cross-validation', )
                elif name == 'classifier':
                    rows = rows + ('Classifier', )
                else:
                    rows = rows + (slice(None), )
            plt.errorbar(
                x=evaluation.loc[rows, ('fit time MEAN')],
                y=evaluation.loc[rows, ('score time MEAN')],
                xerr=evaluation.loc[rows, ('fit time SD')],
                yerr=evaluation.loc[rows, ('score time SD')],
                label=filename,
                marker='x',
                linestyle='')

            idx = idx + 1

        # Plot baselines
        baselines = ['Baseline stratified', 'Baseline uniform']
        for baseline in baselines:
            rows = ()
            for name in evaluation.index.names:
                if name == 'validation method':
                    rows = rows + ('cross-validation', )
                elif name == 'classifier':
                    rows = rows + (baseline, )
                else:
                    rows = rows + (slice(None), )
            plt.errorbar(
                x=evaluation.loc[rows, ('fit time MEAN')],
                y=evaluation.loc[rows, ('score time MEAN')],
                xerr=evaluation.loc[rows, ('fit time SD')],
                yerr=evaluation.loc[rows, ('score time SD')],
                label=baseline,
                marker='x',
                linestyle='')
            idx = idx + 1

    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.title('Efficiency from cross-validation')
    plt.xlabel('fit time')
    plt.ylabel('score time')
    plt.legend()
    plt.grid()
    plt.show()
    fig.savefig(os.path.join(path, 'efficiency.png'), format='png', dpi=200,
        bbox_inches='tight')

    return