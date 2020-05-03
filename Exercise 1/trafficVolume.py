# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:07:04 2020

@author: guser
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

def plotPie(dataFrame):
    labels = dataFrame.astype('category').cat.categories.tolist()
    counts = dataFrame.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)* 100) 

#def mean_root_squared_percentage_error(y_true, y_pred): 
#    y_true, y_pred = np.array(y_true), np.array(y_pred)
#    return np.sqrt(np.mean(np.square((y_true - y_pred)/ y_true)))* 100 

def checkPerformance(y_test,y_pred):
    plt.figure()
    plt.plot(y_test.values,label='true',alpha=0.7)
    plt.plot(y_pred,label='prediction',alpha=0.7)
    plt.xlabel('Samples')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.show()
    
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(y_test, y_pred))
   # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    #print('Root Relative Squared Error:', mean_root_squared_percentage_error(y_test, y_pred))
    print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))
    print('R Squared:', metrics.r2_score(y_test, y_pred))
    
#%% data pre-processing
#http://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume#

#holiday Categorical US National holidays plus regional holiday, Minnesota State Fair
#temp Numeric Average temp in kelvin
#rain_1h Numeric Amount in mm of rain that occurred in the hour
#snow_1h Numeric Amount in mm of snow that occurred in the hour
#clouds_all Numeric Percentage of cloud cover
#weather_main Categorical Short textual description of the current weather
#weather_description Categorical Longer textual description of the current weather
#date_time DateTime Hour of the data collected in local CST time
#traffic_volume Numeric Hourly I-94 ATR 301 reported westbound traffic volume
dataSetPath='./DataSets/Metro_Interstate_Traffic_Volume.csv'

rawData=pd.read_csv(dataSetPath)

#%% investigate data
#plt.figure()
#rawData.boxplot()
#plt.figure()
#rawData['temp'].plot()
#plt.title('temp')
#plt.ylabel('Temperature [°K]')
#plt.xlabel('Index')
#plt.figure()
#
#rawData['rain_1h'].plot()
#plt.title('rain_1h')
#plt.ylabel('Rain per hour [mm/h]')
#plt.xlabel('Index')
#plt.figure()
#rawData['snow_1h'].plot()
#plt.title('snow_1h')

#Correlation Plot
corr = rawData.corr()
plt.figure()
sns.heatmap(corr,annot=True,linewidths=0.5)#,cmap='twilight')
plt.show()
plt.tight_layout()

#visualize categories
plt.figure()
rawData.hist()
plt.show()
plt.tight_layout()

#show box plot of raw data
plt.figure()
plt.subplot(3,2,1)
plt.boxplot(rawData['clouds_all'])
plt.title('clouds_all')

plt.subplot(3,2,2)
plt.boxplot(rawData['rain_1h'])
plt.title('rain_1h')

plt.subplot(3,2,3)
plt.boxplot(rawData['snow_1h'])
plt.title('snow_1h')

plt.subplot(3,2,4)
plt.boxplot(rawData['temp'])
plt.title('temp')

plt.subplot(3,2,5)
plt.boxplot(rawData['traffic_volume'])
plt.title('traffic_volume')

plt.show()
plt.tight_layout()

#% count plots
plt.figure()
sns.countplot(y='weather_main', data=rawData)
plt.tight_layout()

plt.figure()
sns.countplot(y='weather_description', data=rawData)
plt.tight_layout()


plt.figure()
sns.countplot(y='holiday', data= rawData.loc[rawData.holiday != 'None'])
plt.show()
plt.tight_layout()

#change category values to numerical
nbHolidayCat=rawData['holiday'].value_counts().count() #12
nbWeatherMainCat=rawData['weather_main'].value_counts().count()#11
nbWeatherDescription=rawData['weather_description'].value_counts().count() #38

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

X=data.drop(['traffic_volume'],axis=1)
y=data['traffic_volume']

#%%
#pd.scatter_matrix(data)

#%%split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)

#%Data scaling
scaler1 = preprocessing.StandardScaler().fit(X_train)
scaler2 = preprocessing.MinMaxScaler().fit(X_train)

X_train_scaled1=scaler1.transform(X_train)
X_test_scaled1=scaler1.transform(X_test)

X_train_scaled2=scaler2.transform(X_train)
X_test_scaled2=scaler2.transform(X_test)

#%%ridge regression
X_train_RR=X_train_scaled2
X_test_RR=X_test_scaled2
#X_train_RR=X_train
#X_test_RR=X_test
#will be normalized by subtracting mean and dividing by l2-norm
reg = linear_model.Ridge(alpha=100, normalize=False)
reg.fit(X_train_RR,y_train)
y_pred_reg=reg.predict(X_test_RR)

checkPerformance(y_test, y_pred_reg)

#%%KNN
#scaling - makes the reults worse!!??

X_train_knn=X_train_scaled2
X_test_knn=X_test_scaled2
#Without scaling:
#X_train_knn=X_train
#X_test_knn=X_test

knn = KNeighborsRegressor(n_neighbors=5, weights='distance') #distance performs better
knn.fit(X_train_knn,y_train)
y_pred_knn=knn.predict(X_test_knn)

checkPerformance(y_test, y_pred_knn)

#%%Decission Tree Regression

X_train_dt=X_train
X_test_dt=X_test
#X_train_dt=X_train_scaled2
#X_test_dt=X_test_scaled2

dt = tree.DecisionTreeRegressor() #MSE for measuring the quality of the split 



dt.fit(X_train_dt,y_train)



y_pred_dt=dt.predict(X_test_dt)

checkPerformance(y_test, y_pred_dt)

#%% Random Forest
X_train_rf=X_train
X_test_rf=X_test

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train_rf,y_train)
y_pred_rf=rf.predict(X_test_rf)

checkPerformance(y_test, y_pred_rf)



