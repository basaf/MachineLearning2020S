# -*- coding: utf-8 -*-

import configuration as cfg
import os

import functions
import pandas as pd

training_data = pd.read_csv(os.path.join(cfg.default.congressional_voting_data, 'CongressionalVotingID.shuf.train.csv'),
                            sep=',', header=0, index_col=0)

test_data = pd.read_csv(os.path.join(cfg.default.congressional_voting_data, 'CongressionalVotingID.shuf.test.csv'),
                        sep=',', header=0, index_col=0)

training_data_x = training_data.loc[:, training_data.columns != 'class']
training_data_y = training_data.loc[:, 'class']

print("ende")