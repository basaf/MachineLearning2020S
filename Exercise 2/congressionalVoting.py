# -*- coding: utf-8 -*-

import os

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import configuration as cfg
import functions
from joblib import dump, load

print('Start evaulation of Congressional Voting dataset')

training_data = pd.read_csv(os.path.join(cfg.default.congressional_voting_data, 'CongressionalVotingID.shuf.train.csv'),
                            sep=',', header=0, index_col=0)

test_data = pd.read_csv(os.path.join(cfg.default.congressional_voting_data, 'CongressionalVotingID.shuf.test.csv'),
                        sep=',', header=0, index_col=0)

# check if in the training data are empty cells
number_of_empty_cells = training_data.isnull().sum().sum()

if number_of_empty_cells > 0:
    # handle empty cells
    # TODO: implement strategy
    print(f'Number of empty cells {number_of_empty_cells}')

print(f'Number of total samples: {len(training_data) + len(test_data)}')
print(f'Number of training samples: {len(training_data)}')
print(f'Number of test samples: {len(test_data)}')
print(f'Number of features: {len(test_data.columns)}')

unique_classes = training_data['class'].unique()
unique_classes.sort()
print(f'Number of classes: {len(unique_classes)}')
print(f'Classes: {unique_classes}')

# check how the feature values are distributed
for num, column in enumerate(test_data.columns):
    unique_values = training_data[column].unique()
    unique_values.sort()
    print(f'Analyze feature: {column}; unique values: {unique_values}')
    # print(training_data[column].value_counts(normalize=True))

training_data_x = training_data.loc[:, training_data.columns != 'class'].to_numpy()
training_data_y = training_data.loc[:, 'class'].to_numpy()

# handle input features
enc = OrdinalEncoder()
training_data_x_encoded = enc.fit_transform(training_data_x)

# handle output features
le = LabelEncoder()
le.fit(unique_classes)
training_data_y_encoded = le.transform(training_data_y)

# fix the random seed
random_seed = 1
# set the number of test elements to 20%
percentage_test = 0.2

validation_methods = ['holdout', 'cross-validation']
baselines = ['stratified', 'uniform']

path = cfg.default.congressional_voting_figures

# %% k-Nearest Neighbor Classification
# k-nn
if True:
    functions.knn(X=training_data_x_encoded, y=training_data_y_encoded,
                  test_size=percentage_test,
                  random_state=random_seed,
                  list_k=[1, 2, 5, 8, 9, 10, 11, 12, 15, 20],
                  scaling=True,
                  weights=['uniform', 'distance'],
                  validation_methods=validation_methods,
                  baselines=baselines,
                  path=path,
                  filename='knn')

if True:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_knn(path, 'knn')
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_knn(path, 'knn')

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_knn(path, 'knn')
if True:
    # List variants with highest and lowest accuracy values
    filename = 'knn'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'), key='evaluation')

    print('Highest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
                           'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
           sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None), slice(None),
                           'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
           sort_values('accuracy MEAN', ascending=True)).head())

# %% Decision Tree Classification
if True:
    list_max_depth = [1, 10, 100, 1000]
    list_min_samples_split = [2, 20, 200, 2000]
    list_min_samples_leaf = [1, 10, 200, 2000]

    functions.dt(training_data_x_encoded, training_data_y_encoded, percentage_test, random_seed, list_max_depth,
                 list_min_samples_split, list_min_samples_leaf,
                 validation_methods, baselines,
                 path,
                 'dt')

if True:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_dt(path, 'dt')

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_dt(path, 'dt')
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_dt(path, 'dt')

if True:
    # List variants with highest and lowest accuracy values
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

# %% Ridge Classification
if True:
    list_alpha = [0, 1e-4, 1e-2, 1, 5, 10, 50, 100]
    functions.ridge(X=training_data_x_encoded, y=training_data_y_encoded,
                    test_size=percentage_test,
                    random_state=random_seed,
                    list_alpha=list_alpha,
                    scaling=True,
                    validation_methods=['holdout', 'cross-validation'],
                    baselines=['stratified', 'uniform'],
                    path=path,
                    filename='ridge')

if False:
    # Plot performance (efficiency and effectiveness)
    functions.plot_evaluation_ridge(path, 'ridge')
if False:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency_ridge(path, 'ridge')

if False:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy_ridge(path, 'ridge')
if True:
    # List variants with highest and lowest accuracy values

    filename = 'ridge'
    evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'),
                             key='evaluation')

    print('Highest accuracy:')
    print(
        (evaluation.loc[(slice(None), slice(None), 'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
         sort_values('accuracy MEAN', ascending=False)).head())
    print()
    print('Lowest accuracy:')
    print((evaluation.loc[(slice(None), slice(None),
                           'cross-validation', 'Classifier'), ('accuracy MEAN', 'accuracy SD')].
           sort_values('accuracy MEAN', ascending=True)).head())

# %% Compare the different classifiers
filenames = ['knn', 'ridge', 'dt']

if True:
    # For cross-validation scatter-plot accuracy mean and standard deviation
    functions.plot_accuracy(path, filenames)
if True:
    # For cross-validation scatter-plot fit time mean and score time
    functions.plot_efficiency(path, filenames)

if True:
    # List variants of each classifier with highest accuracy values
    all_evaluations = pd.DataFrame()
    for filename in filenames:
        evaluation = pd.read_hdf(os.path.join(path, filename + '_evaluation.h5'), key='evaluation')

        # Select only rows with cross-validation and only Classifier (no baselines)
        rows = ()
        for name in evaluation.index.names:
            if name == 'validation method':
                rows = rows + ('cross-validation',)
            elif name == 'classifier':
                rows = rows + ('Classifier',)
            else:
                rows = rows + (slice(None),)
        evaluation = evaluation.loc[rows, ('accuracy MEAN', 'accuracy SD')]

        # Flatten multiIndex to tuple and add filename
        index = evaluation.index.to_flat_index()
        index_new = [' '.join([filename, str(entry)]) for entry in index]
        evaluation.index = index_new

        # Add current evaluation to overall comparison
        all_evaluations = pd.concat([all_evaluations, evaluation])

    print('Highest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].sort_values(['accuracy MEAN', 'accuracy SD'],
                                                                         ascending=[False, True])).head(10))
    print()
    print('Lowest accuracy:')
    print((all_evaluations[['accuracy MEAN', 'accuracy SD']].sort_values(['accuracy MEAN', 'accuracy SD'],
                                                                         ascending=[True, False])).head(10))

    # Save sorted DataFrame (descending mean value, ascending SD)
    # as csv and h5 files
    filename = '_'.join(['all_evaluation', '_'.join(filenames)])
    ((all_evaluations.sort_values(['accuracy MEAN', 'accuracy SD'],
                                  ascending=[False, True])).to_csv(os.path.join(path, filename + '.csv'), sep=';',
                                                                   decimal=','))
    ((evaluation.sort_values(['accuracy MEAN', 'accuracy SD'],
                             ascending=[False, True])).to_hdf(os.path.join(path, filename + '.h5'), key='evaluation',
                                                              mode='w'))

print()
print('Done')


#%% create kaggle file
if False:
    #load model
    filename='ridge'
    f=os.path.join(path, 'ridge_0_scaling_cross-validation_bestEstimator.joblib')
    model=load(f)
    
    #encode test data
    test_data_encoded = enc.transform(test_data)
    
    #predict
    y_test_data_enc=model.predict(test_data_encoded)
    
    y_test_data=le.inverse_transform(y_test_data_enc)
    #write output file
    kaggleOutput=pd.DataFrame( data={'ID':test_data.index.values,'Class':y_test_data})
    kaggleFile=f=os.path.join(path, 'kaggle.csv')
    kaggleOutput.to_csv(kaggleFile, index=False)