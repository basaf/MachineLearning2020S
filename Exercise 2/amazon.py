# -*- coding: utf-8 -*-

import configuration as cfg
import os

import functions
import pandas as pd
import matplotlib.pyplot as plt

print('Start evaulation of Amazon dataset')

training_data = pd.read_csv(os.path.join(cfg.default.amazon_data, 'amazon_review_ID.shuf.lrn.csv'),
                            sep=',', header=0, index_col=0)

test_data = pd.read_csv(os.path.join(cfg.default.amazon_data, 'amazon_review_ID.shuf.tes.csv'),
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

unique_classes = training_data['Class'].unique()
unique_classes.sort()
print(f'Number of classes: {len(unique_classes)}')
print(f'Classes: {unique_classes}')

# check how the feature values are distributed
for num, column in enumerate(test_data.columns):
    unique_values = training_data[column].unique()
    unique_values.sort()
    print(f'Analyze feature: {column}; unique values: {unique_values}')
    # print(training_data[column].value_counts(normalize=True))

training_data_x = training_data.loc[:, training_data.columns != 'Class'].to_numpy()  # input for classification
training_data_y = training_data.loc[:, 'Class'].to_numpy()  # output class

# fix the random seed
random_seed = 1
# set the number of test elements to 20%
percentage_test = 0.2

# k-nn
functions.knn(X=training_data_x, y=training_data_y, test_size=percentage_test, random_state=random_seed,
              list_k = [1, 5],
              scaling=True,
              weights=['uniform', 'distance'],
              validation_methods=['holdout', 'cross-validation'],
              baselines=['stratified', 'uniform'],
              path=cfg.default.amazon_figures,
              filename='knn')

functions.plot_evaluation_knn(cfg.default.amazon_figures, 'knn')

print('End')
