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
rawData = load(open(os.path.join(cfg.default.dataset_1_path, 'communities.arff'), 'r'))
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
plt.figure()
communities_data[predictive_attributes[0:30]].boxplot()
communities_data[predictive_attributes[30:60]].boxplot()
communities_data[predictive_attributes[60:90]].boxplot()
stophere
plt.figure()
rawData['temp'].plot()
plt.figure()
rawData['rain_1h'].plot()
plt.figure()
rawData['snow_1h'].plot()

#change category values to numerical
nbHolidayCat=rawData['holiday'].value_counts().count() #12
nbWeatherMainCat=rawData['weather_main'].value_counts().count()#11
nbWeatherDescription=rawData['weather_description'].value_counts().count() #38

#%%Data scaling

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