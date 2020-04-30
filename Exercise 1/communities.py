# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:31:00 2020

@author: Steindl, Windholz
"""
from arff import load
import configuration as cfg
import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from seaborn import heatmap

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import neural_network

from sklearn import preprocessing

def plotPie(dataFrame):
    labels = dataFrame.astype('category').cat.categories.tolist()
    counts = dataFrame.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.show()

def checkPerformance(y_test,y_pred):
    plt.figure()
    plt.plot(y_test.values,label='true')
    plt.plot(y_pred,label='y_hat')
    plt.legend()
    plt.show()
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))
    
#%% data pre-processing
# load dataset (.arff) into pandas DataFrame
rawData = load(open(os.path.join(cfg.default.communities_data,
                                 'communities.arff'), 'r'))
all_attributes = list(i[0] for i in rawData['attributes'])
communities_data = pd.DataFrame(columns=all_attributes, data=rawData['data'])

# divide attributes in not_predictive, predictive and goal
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

#%% investigate data
if False:
    communities_data[predictive_attributes[0:30]].boxplot()
    communities_data[predictive_attributes[30:60]].boxplot()
    communities_data[predictive_attributes[60:90]].boxplot()

# plt.show()
# plt.figure()
# rawData['temp'].plot()
# plt.figure()
# rawData['rain_1h'].plot()
# plt.figure()
# rawData['snow_1h'].plot()


#%% Treat missing values
missing_values = (communities_data[predictive_attributes+[goal_attribute]].
                  isnull().sum().sum())
cells_total = (len(communities_data.index)*
    len(communities_data[predictive_attributes+[goal_attribute]].columns))
print('Missing values: '+str(missing_values))
print('Cells total: '+str(cells_total))
print('Missing: {:.1%}'.format(missing_values/cells_total))

# Remove attributes with more than 80 % missing values
attributes_to_delete = communities_data[predictive_attributes].columns[
    communities_data[predictive_attributes].isnull().sum() / 
    len(communities_data.index)*100 > 80]
for x in attributes_to_delete:
    predictive_attributes.remove(x)

# Impute mean value of attribute using sklearn (even if pandas would be faster)
print('Missing in "OtherPerCap": '+
      str(communities_data['OtherPerCap'].isnull().sum()))
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
communities_data['OtherPerCap'] = imp.fit_transform(
    communities_data['OtherPerCap'].to_numpy().reshape(-1, 1)).flatten()

# Input variable correlation analysis
correlation_matrix = (communities_data[predictive_attributes+[goal_attribute]].
                      corr(method='pearson'))

if False:
    ax = heatmap(correlation_matrix, center=0, vmin=-1, vmax=1, square=True,
                xticklabels=False, yticklabels=False)
    # plt.gcf().subplots_adjust(bottom=0.48, left=0.27, right=0.99, top=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.default.dataset_1_figures_path, 
                'communities_data_correlations.png'),
                format='png', dpi=200,
                metadata={'Creator': '', 'Author': '', 'Title': '', 'Producer': ''},
                )

#%% Data encoding
# Not necessary 

#%% Split data
X = communities_data[predictive_attributes]
y = communities_data[goal_attribute] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Data scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

stophere

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
data=data.drop(['weather_description'],axis=1)
data=pd.get_dummies(data, columns=['weather_main'], prefix = ['weatherMain'])

X=data.drop(['traffic_volume'],axis=1)
y=data['traffic_volume']

#%%split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
#%%ridge regression

#will be normalized by subtracting mean and dividing by l2-norm
reg = linear_model.Ridge(alpha=.5)#,normalize=True)
reg.fit(X_train,y_train)
y_pred_reg=reg.predict(X_test)

checkPerformance(y_test, y_pred_reg)

#%%KNN
#scaling - makes the reults worse!!??

X_train_knn=X_train_scaled
X_test_knn=X_test_scaled
#Without scaling:
#X_train_knn=X_train
#X_test_knn=X_test

knn = KNeighborsRegressor(n_neighbors=5, weights='distance') #distance performs better
knn.fit(X_train_knn,y_train)
y_pred_knn=knn.predict(X_test_knn)

checkPerformance(y_test, y_pred_knn)

#%%Decission Tree Regression

dt = tree.DecisionTreeRegressor() #MSE for measuring the quality of the split 
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)

checkPerformance(y_test, y_pred_dt)

#%%Multi-layer Perceptron
X_train_mlp=X_train_scaled
X_test_mlp=X_test_scaled
mlp=neural_network.MLPRegressor(solver='adam', hidden_layer_sizes=(50,10), max_iter=400,verbose=True)
mlp.fit(X_train_mlp,y_train)
y_pred_mlp=mlp.predict(X_test_mlp)

checkPerformance(y_test, y_pred_mlp)