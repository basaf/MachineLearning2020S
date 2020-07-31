# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from arff import load

import configuration as cfg


def preprocess_communities():
    # load dataset (.arff) into pandas DataFrame
    rawData = load(open(os.path.join(cfg.default.communities_data, 'communities.arff'), 'r'))
    all_attributes = list(i[0] for i in rawData['attributes'])
    communities_data = pd.DataFrame(columns=all_attributes, data=rawData['data'])

    # distinguish attributes in not_predictive, predictive and goal
    not_predictive_attributes = [
        'state',
        'county',
        'community',
        'communityname',
        'fold'
    ]
    goal_attribute = 'ViolentCrimesPerPop'

    predictive_attributes = all_attributes.copy()
    predictive_attributes.remove(goal_attribute)

    for x in not_predictive_attributes:
        predictive_attributes.remove(x)

    # Treat missing values
    missing_values = (communities_data[predictive_attributes + [goal_attribute]].
                      isnull().sum().sum())
    cells_total = (len(communities_data.index) *
                   len(communities_data[predictive_attributes + [goal_attribute]].columns))
    print('Missing values: ' + str(missing_values))
    print('Cells total: ' + str(cells_total))
    print('Missing: {:.1%}'.format(missing_values / cells_total))

    # Remove attributes with more than 80 % missing values
    attributes_to_delete = communities_data[predictive_attributes].columns[
        communities_data[predictive_attributes].isnull().sum() /
        len(communities_data.index) * 100 > 80]
    for x in attributes_to_delete:
        predictive_attributes.remove(x)

    print('Missing in "OtherPerCap": ' +
          str(communities_data['OtherPerCap'].isnull().sum()))

    # Split data
    X = communities_data[predictive_attributes].to_numpy()

    # -> impute mean value of attribute, but do the split before
    col_mean = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    X[idx_nan] = np.take(col_mean, idx_nan[1])

    y = communities_data[goal_attribute].to_numpy()

    return X, y


def preprocess_traffic():
    rawData = pd.read_csv(os.path.join(cfg.default.traffic_data,
                                       'Metro_Interstate_Traffic_Volume.csv'))
    # remove outliers
    # remove temperatrues with 0°K
    rawData['temp'].replace(0, np.NaN, inplace=True)

    rawData.loc[rawData['rain_1h'] > 9000, 'rain_1h'] = np.NaN

    rawData = rawData.dropna()

    # Data encoding
    # first attempt:
    # make holiday binary - because their are only a few days in the data set (reduce curse of dimensionality)
    data = rawData.copy()
    data.loc[data.holiday == 'None', 'holiday'] = 0
    data.loc[data.holiday != 0, 'holiday'] = 1

    # use month and weekday and hour of day as input with simple label encoding
    data['date_time'] = pd.to_datetime(data['date_time'])

    data.insert(8, 'month', data['date_time'].dt.month)
    data.insert(9, 'dayOfWeek', data['date_time'].dt.dayofweek)
    data.insert(10, 'hourOfDay', data['date_time'].dt.hour)
    data = data.drop(['date_time'], axis=1)

    # ignore weather description and only use weatherMain with hotEncoding
    data = data.drop(['weather_description'], axis=1)

    # one-hot-encoding
    data = pd.get_dummies(data, columns=['weather_main'], prefix=['weatherMain'])

    X = data.drop(['traffic_volume'], axis=1).to_numpy()
    y = data['traffic_volume'].to_numpy()

    return X, y


def preprocess_real_estate():
    rawData = pd.read_excel(os.path.join(cfg.default.real_estate_data,
                                         'Real estate valuation data set.xlsx'))

    # change the transaction date to year, and month field
    transactionDate = rawData['X1 transaction date']

    transactionMonth = \
        ((rawData['X1 transaction date'] - rawData['X1 transaction date'].astype(int)) / (1 / 12)).astype(int)
    transactionYear = rawData['X1 transaction date'].astype(int)

    data = rawData.copy()
    data.drop('X1 transaction date', axis=1, inplace=True)
    data['X1 transaction year'] = transactionYear.values
    data['X1 transaction month'] = transactionMonth.values

    X = data.drop(['Y house price of unit area'], axis=1).to_numpy()
    y = data['Y house price of unit area'].to_numpy()

    return X, y


def preprocess_student():
    rawData = pd.read_csv(os.path.join(cfg.default.student_data,
                                       'student-mat.csv'), sep=';')

    categorial = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                  'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                  'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                  'romantic']

    for feature in categorial:
        rawData[feature] = rawData[feature].astype('category')
        rawData[feature] = rawData[feature].cat.codes

    x = rawData.to_numpy()

    data = rawData.copy()

    X = data.drop(['G3'], axis=1)
    y = data['G3']

    X = X.to_numpy()
    y = y.to_numpy()

    return X, y
