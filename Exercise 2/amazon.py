# -*- coding: utf-8 -*-

import configuration as cfg
import os

import functions
import pandas as pd

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

training_data_x = training_data.loc[:, training_data.columns != 'Class']  # input for classification
training_data_y = training_data.loc[:, 'Class']  # output class

print('ende')
