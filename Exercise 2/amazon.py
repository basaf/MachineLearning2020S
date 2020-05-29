# -*- coding: utf-8 -*-

import configuration as cfg
import os

import functions
import pandas as pd

training_data = pd.read_csv(os.path.join(cfg.default.amazon_data, 'amazon_review_ID.shuf.lrn.csv'),
                            sep=',', header=0, index_col=0)

test_data = pd.read_csv(os.path.join(cfg.default.amazon_data, 'amazon_review_ID.shuf.tes.csv'),
                        sep=',', header=0, index_col=0)

training_data_x = training_data.loc[:, training_data.columns != 'Class']
training_data_y = training_data.loc[:, 'Class']

print('ende')