# -*- coding: utf-8 -*-

from arff import load
import configuration as cfg
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap

from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import helper
import functions

#%% data pre-processing

occupancy_TrainingData = pd.read_csv(
    os.path.join(cfg.default.occupancy_data, 'datatraining.txt'))
occupancy_TrainingData = occupancy_TrainingData.set_index('date')
# occupancy_TrainingData['HumidityRatio g/kg'] = (
#     occupancy_TrainingData['HumidityRatio']*1000 )

occupancy_test_data = pd.read_csv(
    cfg.default.occupancy_data + '\\datatest.txt')
occupancy_test_data = occupancy_test_data.set_index('date')
# occupancy_test_data['HumidityRatio g/kg'] = (
#     occupancy_test_data['HumidityRatio']*1000 )

occupancy_test2_data = pd.read_csv(
    cfg.default.occupancy_data + '\\datatest2.txt')
occupancy_test2_data = occupancy_test2_data.set_index('date')
# occupancy_test2_data['HumidityRatio g/kg'] = (
#     occupancy_test2_data['HumidityRatio']*1000 )

# Concatenate data sets
data = pd.concat([occupancy_TrainingData,
    occupancy_test_data,
    occupancy_test2_data],
    sort=True, verify_integrity=True)

# distinguish attributes in predictive and goal
goal_attribute = 'Occupancy'

predictive_attributes = data.columns.to_list()
predictive_attributes.remove(goal_attribute)

#%% Investigate data
if False:
    helper.boxplot_raw_data(data,
        data[predictive_attributes].columns,
        save_fig_path=os.path.join(cfg.default.occupancy_figures,
            'occupancy_box_plot.png'))

#%% Treat missing values
missing_values = (data[predictive_attributes+[goal_attribute]].
                  isnull().sum().sum())
cells_total = (len(data.index)*
    len(data[predictive_attributes+[goal_attribute]].columns))
print('Missing values: '+str(missing_values))
print('Cells total: '+str(cells_total))
print('Missing: {:.1%}'.format(missing_values/cells_total))

# Remove attributes with more than 80 % missing values
if False:
    attributes_to_delete = data[predictive_attributes].columns[
        data[predictive_attributes].isnull().sum() / 
        len(data.index)*100 > 80]
    for x in attributes_to_delete:
        predictive_attributes.remove(x)

#%% Data encoding
# use month and weekday and hour of day as input with simple label encoding
data['dayOfWeek'] = pd.to_datetime(data.index).dayofweek
data['hourOfDay'] = pd.to_datetime(data.index).hour
predictive_attributes.append('dayOfWeek')
predictive_attributes.append('hourOfDay')

#%% Variable correlation analysis
correlation_matrix = (data[predictive_attributes+[goal_attribute]].
                      corr(method='pearson'))
if False:
    ax = heatmap(correlation_matrix, center=0, vmin=-1, vmax=1, square=True, 
        xticklabels=True, yticklabels=True, annot=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 
        horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.occupancy_figures,
        'occupancy_data_correlations.png'),
        format='png', dpi=200,
        metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})

# Plot and save histogram
if True:
    data[predictive_attributes+[goal_attribute]].hist()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.occupancy_figures, 'occupancy_all_data_histogram.png'), format='png',
                metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''})
    plt.close()

#%% Split data
X = data[predictive_attributes].to_numpy()
y = data[goal_attribute].to_numpy() 
test_size = 0.2
random_state = 1
# Splitting is done in sub routines, since whole data set is used for k-fold CV
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#     random_state=1)

#%% Impute mean value of attributes
if False:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

#%% Data scaling (remove mean and scale to unit variance)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)


validation_methods = ['holdout', 'cross-validation']
baselines=['stratified', 'uniform']

#%% k-Nearest Neighbor Classification
if False:
    list_k = [1, 10, 50, 100, 300, 500]
    weights = ['uniform', 'distance']

    functions.knn(X, y, test_size, random_state, list_k, True,
        weights, validation_methods, baselines,
        cfg.default.occupancy_figures,
        'knn')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_knn(cfg.default.occupancy_figures, 'knn')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_knn(cfg.default.occupancy_figures, 'knn')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_knn(cfg.default.occupancy_figures, 'knn')
if False:
    # List variants with highest and lowest accuracy values
    path = cfg.default.occupancy_figures
    filename = 'knn'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())


#%% Naïve Bayes Classification
if False:
    # Scaling not needed for algorithms that don’t use distances like Naive
    # Bayes
    functions.gnb(X, y, test_size, random_state,
        validation_methods, baselines,
        cfg.default.occupancy_figures,
        'gnb')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_gnb(cfg.default.occupancy_figures, 'gnb')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_gnb(cfg.default.occupancy_figures, 'gnb')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_gnb(cfg.default.occupancy_figures, 'gnb')

#%% Decision Tree Classification
if False:
    list_max_depth = [1, 10, 100, 1000]
    list_min_samples_split = [2, 20, 200, 2000]
    list_min_samples_leaf = [1, 10, 200, 2000]

    functions.dt(X, y, test_size, random_state, list_max_depth,
        list_min_samples_split, list_min_samples_leaf,
        validation_methods, baselines,
        cfg.default.occupancy_figures,
        'dt')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_dt(cfg.default.occupancy_figures, 'dt')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_dt(cfg.default.occupancy_figures, 'dt')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_dt(cfg.default.occupancy_figures, 'dt')

if False:
    # List variants with highest and lowest accuracy values
    path = cfg.default.occupancy_figures
    filename = 'dt'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())

#%% Ridge Classification
if False:
    list_alpha = [0, 1e-4, 1e-2, 1, 5, 10, 50, 100]
    functions.ridge(X, y, test_size, random_state, list_alpha, True,
        validation_methods, baselines,
        cfg.default.occupancy_figures,
        'ridge')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_ridge(cfg.default.occupancy_figures, 'ridge')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_ridge(cfg.default.occupancy_figures, 'ridge')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_ridge(cfg.default.occupancy_figures, 'ridge')
if False:
    # List variants with highest and lowest accuracy values
    path = cfg.default.occupancy_figures
    filename = 'ridge'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
        key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None),
        'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
        sort_values('accuracy MEAN', ascending=True)).head())

#%% Compare the different classifiers 
filenames = ['knn', 'ridge', 'dt']
if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy(cfg.default.occupancy_figures, filenames)
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency(cfg.default.occupancy_figures, filenames)

if False:
    # List variants of each classifier with highest accuracy values
    all_evaluations = pd.DataFrame()
    for filename in filenames:
        path = cfg.default.occupancy_figures
        evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
            key='evaluation')

        # Select only rows with cross-validation and only Classifier (no baselines)
        rows = ()
        for name in evaluation.index.names:
            if name == 'validation method':
                rows = rows + ('cross-validation', )
            elif name == 'classifier':
                rows = rows + ('Classifier', )
            else:
                rows = rows + (slice(None), )
        evaluation = evaluation.loc[rows, ('accuracy MEAN', 'accuracy SD')]

        # Flatten multiIndex to tuple and add filename
        index = evaluation.index.to_flat_index()
        index_new = [' '.join([filename, str(entry)]) for entry in index]
        evaluation.index = index_new

        # Add current evaluation to overall comparison
        all_evaluations = pd.concat([all_evaluations, evaluation])

    print('Highest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].
        sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).head(10))
    print()
    print('Lowest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].
        sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[True, False])).head(10))

    # Save sorted DataFrame (descending mean value, ascending SD)
    # as csv and h5 files
    filename = '_'.join(['all_evaluation', '_'.join(filenames)])
    ((all_evaluations.sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).to_csv(os.path.join(path, filename + '.csv'), sep=';', decimal=','))
    ((evaluation.sort_values(['accuracy MEAN', 'accuracy SD'],
            ascending=[False, True])).to_hdf(os.path.join(path, filename + '.h5'), key='evaluation', mode='w'))



print()
print('Done')

