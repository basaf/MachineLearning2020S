# -*- coding: utf-8 -*-

import configuration as cfg
import os

import functions
import pandas as pd

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

training_data_x = training_data.loc[:, training_data.columns != 'class']
training_data_y = training_data.loc[:, 'class']

print('End')
