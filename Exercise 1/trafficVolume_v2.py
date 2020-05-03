# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:26:31 2020

@author: Pannosch, Steindl, Windholz
"""

# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import configuration as cfg
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

import helper
import functions

import seaborn as sns
from seaborn import heatmap

import numpy as np

dataSetPath='./DataSets/Metro_Interstate_Traffic_Volume.csv'
rawData=pd.read_csv(dataSetPath)


#investigate data

#%% box plot
helper.boxplot_raw_data(rawData, rawData.columns[[1, 2, 3, 4, 8]],
                        save_fig_path=os.path.join(cfg.default.traffic_figures, 'traffic_volume_box_plot.png'))


#%% plt correlation matrix
plt.figure()
correlation_matrix = (rawData.loc[:, rawData.columns[[1, 2, 3, 4, 8]]].corr(method='pearson'))

ax = heatmap(correlation_matrix, xticklabels=correlation_matrix.columns,
             yticklabels=correlation_matrix.columns, annot=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.traffic_figures, 'traffic_volume_corr.png'), format='png')
plt.close()

#%% plot histogram
figure = plt.figure()
rawData.hist()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.traffic_figures, 'traffic_volume_hist.png'), format='png')
plt.close(figure)

#%% count plots of categorical
plt.figure()
sns.countplot(y='weather_main', data=rawData)
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.traffic_figures, 'weather_main_count.png'), format='png')
plt.close(figure)

plt.figure()
sns.countplot(y='weather_description', data=rawData)
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.traffic_figures, 'weather_description_count.png'), format='png')
plt.close(figure)


plt.figure()
sns.countplot(y='holiday', data= rawData.loc[rawData.holiday != 'None'])
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(cfg.default.traffic_figures, 'holiday_count.png'), format='png')
plt.close(figure)


#%%
#%% remove outliers
#remove temperatrues with 0°K
rawData['temp'].replace(0,np.NaN, inplace=True)

rawData.loc[rawData['rain_1h'] > 9000, 'rain_1h']=np.NaN

rawData=rawData.dropna()

#%% Data encoding
#first attempt:
#make holiday binary - because ther only a few days in the data set (reduce curse of dimensionality)
data=rawData.copy()
data.loc[data.holiday == 'None', 'holiday']=0
data.loc[data.holiday != 0, 'holiday']=1


#use month and weekday and hour of day as input with simple label encoding
data['date_time']=pd.to_datetime(data['date_time'])

data.insert(8,'month',data['date_time'].dt.month)
data.insert(9,'dayOfWeek',data['date_time'].dt.dayofweek)
data.insert(10,'hourOfDay',data['date_time'].dt.hour)
data=data.drop(['date_time'],axis=1)



#ignore weather description and only use weatherMain with hotEncoding
#ignore
data=data.drop(['weather_description'],axis=1)
#one-hot-encoding
#data=pd.get_dummies(data, columns=['weather_description'], prefix = ['weather_description'])
#label encoding
#data['weather_description']=data['weather_description'].astype('category')
#data['weather_description']=data['weather_description'].cat.codes

#one-hot-encoding
data=pd.get_dummies(data, columns=['weather_main'], prefix = ['weatherMain'])
#or label-encoding
#data['weather_main']=data['weather_main'].astype('category')
#data['weather_main']=data['weather_main'].cat.codes

X=data.drop(['traffic_volume'],axis=1).to_numpy()
y=data['traffic_volume'].to_numpy()

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#%%
print('Ridge Linear Regression')

alpha_list = [0, 0.01,0.1, 1, ]
#alpha_list = [ 1,  10]
functions.ridge_regression(X_train, X_test, y_train, y_test, alpha_list, True,
                           cfg.default.traffic_figures, 'ridge_reg')

#%%
print('KNN')

#k_values = [1, 7]
k_values = [1, 3, 5, 7, 10]

functions.knn(X_train, X_test, y_train, y_test, k_values, True, ['uniform', 'distance'],
              cfg.default.traffic_figures, 'knn')
#%%
print('Decission Tree Regression')

max_depths = [50, 100, 300, 400]

min_weight_fraction_leafs = [.0, .125, .25, .375, .5]

min_samples_leaf=[1, 10, 50, 100]


functions.decision_tree(X_train, X_test, y_train, y_test, max_depths, min_weight_fraction_leafs, min_samples_leaf,
                        cfg.default.traffic_figures, 'dtree')

#%%
print('MLP')

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

max_iteration = 800
solver = 'adam' # lbfgs, adam, sgd
alpha = [0.01,0.001,0.0001]

list_hidden_layer_sizes = [[40],[10,10], [60, 20]]

functions.mlp(X_train_scaled, X_test_scaled, y_train, y_test, max_iteration, solver, alpha, list_hidden_layer_sizes,
        cfg.default.traffic_figures, 'mlp')