# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:37:39 2022

@author: sachi
"""

#import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
import streamlit as st

all_stocks = pd.read_csv("C:/Users/sachi/OneDrive/Desktop/Data Mining/S&P 500 Stocks Data/sp500_stocks.csv")
sector = pd.read_csv("C:/Users/sachi/OneDrive/Desktop/Data Mining/S&P 500 Stocks Data/sp500_companies.csv")

# Join the Sector dataset to all_stocks using left join
all_stocks = all_stocks.merge(sector[['Symbol','Sector']], how='left', on='Symbol')

all_stocks[all_stocks['Symbol'] == 'GOOG']

# Fill the sector values of missing symbols
all_stocks.loc[all_stocks['Symbol']=='GEN','Sector'] = 'Technology'
all_stocks.loc[all_stocks['Symbol']=='ELV','Sector'] = 'Healthcare'
all_stocks.loc[all_stocks['Symbol']=='META','Sector'] = 'Communication Services'
all_stocks.loc[all_stocks['Symbol']=='PARA','Sector'] = 'Communication Services'
all_stocks.loc[all_stocks['Symbol']=='SBUX','Sector'] = 'Consumer Cyclical'
all_stocks.loc[all_stocks['Symbol']=='V','Sector'] = 'Financial Services'
all_stocks.loc[all_stocks['Symbol']=='WBD','Sector'] = 'Communication Services'
all_stocks.loc[all_stocks['Symbol']=='WTW','Sector'] = 'Financial Services'


all_stocks = all_stocks.sort_values(['Symbol','Date']).reset_index(drop=True)
all_stocks['adj_close_lag1'] = all_stocks[['Symbol','Date','Adj Close']].groupby(['Symbol']).shift(1)['Adj Close'].reset_index(drop=True)
all_stocks['return'] = np.log(all_stocks['Adj Close'] / all_stocks['adj_close_lag1'])


# Define a new function called create_lagged_features which accepts two elements a dataframe and a variable.
def create_lagged_features(df, var):
    df[var+'_lag1'] = df[['Symbol','Date',var]].groupby(['Symbol']).shift(1)[var].reset_index(drop=True)
    df[var+'_rolling5'] = df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(5).sum().reset_index(drop=True)
    df[var+'_rolling15'] = df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(15).sum().reset_index(drop=True)
    return df


all_stocks = create_lagged_features(all_stocks, 'return')
all_stocks = create_lagged_features(all_stocks, 'Volume')

# Find the relative volume of Volume_lag1 to Volume_rolling15
all_stocks['relative_vol_1_15'] = all_stocks['Volume_lag1'] / all_stocks['Volume_rolling15']
# Find the relative volume of Volume_rolling5 to Volume_rolling15
all_stocks['relative_vol_5_15'] = all_stocks['Volume_rolling5'] / all_stocks['Volume_rolling15']

sector_counts=all_stocks['Sector'].value_counts()

# perform frequency based encoding (usually this would only use training portion to fit transform, but need to keep transform constant across days)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories = [list(sector_counts.index)])

#create new feature called 'Sector_enc' which fits all 'Sector' values using Ordinal Encoder
all_stocks['Sector_enc'] = enc.fit_transform(all_stocks[['Sector']])


this_stock = 'AAPL'       # I pick apple stock to see what might be driving for that stock based on modeling the market
# I choose this feature_list based on the feature importance and model interpretability
feature_list = ['Adj Close', 'Volume', 'adj_close_lag1', 'return_lag1','return_rolling5','return_rolling15',
                'relative_vol_1_15','relative_vol_5_15', 'Sector_enc']
this_date1 = all_stocks.loc[all_stocks.index[-1],'Date'] 
this_date2 = all_stocks.loc[all_stocks.index[-2],'Date']
this_date3 = all_stocks.loc[all_stocks.index[-3],'Date']
this_date4 = all_stocks.loc[all_stocks.index[-4],'Date']
this_date5 = all_stocks.loc[all_stocks.index[-5],'Date']
this_date6 = all_stocks.loc[all_stocks.index[-6],'Date']
this_date7 = all_stocks.loc[all_stocks.index[-7],'Date']
this_date8 = all_stocks.loc[all_stocks.index[-8],'Date']
this_date9 = all_stocks.loc[all_stocks.index[-9],'Date']
this_date10 = all_stocks.loc[all_stocks.index[-10],'Date']

#create a list of today's stocks EXCLUDING the one we are interested in
stocks_1 = all_stocks[np.logical_and(all_stocks['Date']==this_date1,all_stocks['Symbol']!=this_stock)]  #11/23/2022
stocks_2 = all_stocks[np.logical_and(all_stocks['Date']==this_date2,all_stocks['Symbol']!=this_stock)]  #11/22/2022
stocks_3 = all_stocks[np.logical_and(all_stocks['Date']==this_date3, all_stocks['Symbol']!=this_stock)] #11/21/2022
stocks_4 = all_stocks[np.logical_and(all_stocks['Date']==this_date4,all_stocks['Symbol']!=this_stock)]  #11/18/2022
stocks_5 = all_stocks[np.logical_and(all_stocks['Date']==this_date5, all_stocks['Symbol']!=this_stock)] #11/17/2022
stocks_6 = all_stocks[np.logical_and(all_stocks['Date']==this_date6,all_stocks['Symbol']!=this_stock)]  #11/16/2022
stocks_7 = all_stocks[np.logical_and(all_stocks['Date']==this_date7, all_stocks['Symbol']!=this_stock)] #11/15/2022
stocks_8 = all_stocks[np.logical_and(all_stocks['Date']==this_date8,all_stocks['Symbol']!=this_stock)]  #11/14/2022
stocks_9 = all_stocks[np.logical_and(all_stocks['Date']==this_date9, all_stocks['Symbol']!=this_stock)] #11/11/2022
stocks_10 = all_stocks[np.logical_and(all_stocks['Date']==this_date10,all_stocks['Symbol']!=this_stock)]#11/10/2022


# create a train/test split for early stopping for each day.
X_train1, X_test1, y_train1, y_test1 = train_test_split(stocks_1[feature_list], stocks_1['return'], test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(stocks_2[feature_list], stocks_2['return'], test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(stocks_3[feature_list], stocks_3['return'], test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4 = train_test_split(stocks_4[feature_list], stocks_4['return'], test_size=0.2, random_state=42)
X_train5, X_test5, y_train5, y_test5 = train_test_split(stocks_5[feature_list], stocks_5['return'], test_size=0.2, random_state=42)
X_train6, X_test6, y_train6, y_test6 = train_test_split(stocks_6[feature_list], stocks_6['return'], test_size=0.2, random_state=42)
X_train7, X_test7, y_train7, y_test7 = train_test_split(stocks_7[feature_list], stocks_7['return'], test_size=0.2, random_state=42)
X_train8, X_test8, y_train8, y_test8 = train_test_split(stocks_8[feature_list], stocks_8['return'], test_size=0.2, random_state=42)
X_train9, X_test9, y_train9, y_test9 = train_test_split(stocks_9[feature_list], stocks_9['return'], test_size=0.2, random_state=42)
X_train10, X_test10, y_train10, y_test10 = train_test_split(stocks_10[feature_list], stocks_10['return'], test_size=0.2, random_state=42)


param_grid = {'max_depth':list(range(3,7,1))}
params_fit1 = {"eval_metric" : "mae",'eval_set': [[X_test1, y_test1]],'early_stopping_rounds' : 10}
params_fit2= {"eval_metric" : "mae",'eval_set': [[X_test2, y_test2]],'early_stopping_rounds' : 10}
params_fit3 = {"eval_metric" : "mae",'eval_set': [[X_test3, y_test3]],'early_stopping_rounds' : 10}
params_fit4 = {"eval_metric" : "mae",'eval_set': [[X_test4, y_test4]],'early_stopping_rounds' : 10}
params_fit5 = {"eval_metric" : "mae",'eval_set': [[X_test5, y_test5]],'early_stopping_rounds' : 10}
params_fit6 = {"eval_metric" : "mae",'eval_set': [[X_test6, y_test6]],'early_stopping_rounds' : 10}
params_fit7 = {"eval_metric" : "mae",'eval_set': [[X_test7, y_test7]],'early_stopping_rounds' : 10}
params_fit8 = {"eval_metric" : "mae",'eval_set': [[X_test8, y_test8]],'early_stopping_rounds' : 10}
params_fit9 = {"eval_metric" : "mae",'eval_set': [[X_test9, y_test9]],'early_stopping_rounds' : 10}
params_fit10 = {"eval_metric" : "mae",'eval_set': [[X_test10, y_test10]],'early_stopping_rounds' : 10}


gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

# I choose Grid search because it manually specified subset of the hyperparameter space of the targeted algorithm.
search1 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)   
search1.fit(X_train1,y_train1,**params_fit1)   
search2 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search2.fit(X_train2, y_train2,**params_fit2)
search3 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search3.fit(X_train3, y_train3,**params_fit3)
search4 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search4.fit(X_train4, y_train4,**params_fit4)
search5 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search5.fit(X_train5, y_train5,**params_fit5)
search6 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search6.fit(X_train6, y_train6,**params_fit6)
search7 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search7.fit(X_train7, y_train7,**params_fit7)
search8 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search8.fit(X_train8, y_train8,**params_fit8)
search9 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search9.fit(X_train9, y_train9,**params_fit9)
search10 = GridSearchCV(gbm,param_grid = param_grid, verbose = 1)
search10.fit(X_train10, y_train10,**params_fit10)


a1 = search1.best_estimator_.feature_importances_
a2 = search2.best_estimator_.feature_importances_
a3 = search3.best_estimator_.feature_importances_
a4 = search4.best_estimator_.feature_importances_
a5 = search5.best_estimator_.feature_importances_
a6 = search6.best_estimator_.feature_importances_
a7 = search7.best_estimator_.feature_importances_
a8 = search8.best_estimator_.feature_importances_
a9 = search9.best_estimator_.feature_importances_
a10 = search10.best_estimator_.feature_importances_

avg_feature_importance = (a1+a2+a3+a4+a5+a6+a7+a8+a9+a10)/10
shap.initjs()

# Actual and Predicted return for day 1
this_data1 = all_stocks[np.logical_and(all_stocks['Date'] == this_date1,all_stocks['Symbol']==this_stock)][feature_list]
this_actual1 = all_stocks[np.logical_and(all_stocks['Date']==this_date1,all_stocks['Symbol']==this_stock)]['return']
search1.best_estimator_.predict(this_data1), this_actual1
# Actual and Predicted return for day 2
this_data2 = all_stocks[np.logical_and(all_stocks['Date'] == this_date2,all_stocks['Symbol']==this_stock)][feature_list]
this_actual2 = all_stocks[np.logical_and(all_stocks['Date']==this_date2,all_stocks['Symbol']==this_stock)]['return']
search2.best_estimator_.predict(this_data2), this_actual2
# Actual and Predicted return for day 3
this_data3 = all_stocks[np.logical_and(all_stocks['Date'] == this_date3,all_stocks['Symbol']==this_stock)][feature_list]
this_actual3 = all_stocks[np.logical_and(all_stocks['Date']==this_date3,all_stocks['Symbol']==this_stock)]['return']
search3.best_estimator_.predict(this_data3), this_actual3
# Actual and Predicted return for day 4
this_data4 = all_stocks[np.logical_and(all_stocks['Date'] == this_date4,all_stocks['Symbol']==this_stock)][feature_list]
this_actual4 = all_stocks[np.logical_and(all_stocks['Date']==this_date4,all_stocks['Symbol']==this_stock)]['return']
search4.best_estimator_.predict(this_data4), this_actual4
# Actual and Predicted return for day 5
this_data5 = all_stocks[np.logical_and(all_stocks['Date'] == this_date5,all_stocks['Symbol']==this_stock)][feature_list]
this_actual5 = all_stocks[np.logical_and(all_stocks['Date']==this_date5,all_stocks['Symbol']==this_stock)]['return']
search5.best_estimator_.predict(this_data5), this_actual5
# Actual and Predicted return for day 6
this_data6 = all_stocks[np.logical_and(all_stocks['Date'] == this_date6,all_stocks['Symbol']==this_stock)][feature_list]
this_actual6 = all_stocks[np.logical_and(all_stocks['Date']==this_date6,all_stocks['Symbol']==this_stock)]['return']
search6.best_estimator_.predict(this_data6), this_actual6
# Actual and Predicted return for day 7
this_data7 = all_stocks[np.logical_and(all_stocks['Date'] == this_date7,all_stocks['Symbol']==this_stock)][feature_list]
this_actual7 = all_stocks[np.logical_and(all_stocks['Date']==this_date7,all_stocks['Symbol']==this_stock)]['return']
search7.best_estimator_.predict(this_data7), this_actual7
# Actual and Predicted return for day 8
this_data8 = all_stocks[np.logical_and(all_stocks['Date'] == this_date8,all_stocks['Symbol']==this_stock)][feature_list]
this_actual8 = all_stocks[np.logical_and(all_stocks['Date']==this_date8,all_stocks['Symbol']==this_stock)]['return']
search8.best_estimator_.predict(this_data8), this_actual8
# Actual and Predicted return for day 9
this_data9 = all_stocks[np.logical_and(all_stocks['Date'] == this_date9,all_stocks['Symbol']==this_stock)][feature_list]
this_actual9 = all_stocks[np.logical_and(all_stocks['Date']==this_date9,all_stocks['Symbol']==this_stock)]['return']
search9.best_estimator_.predict(this_data9), this_actual9
# Actual and Predicted return for day 10
this_data10 = all_stocks[np.logical_and(all_stocks['Date'] == this_date10,all_stocks['Symbol']==this_stock)][feature_list]
this_actual10 = all_stocks[np.logical_and(all_stocks['Date']==this_date10,all_stocks['Symbol']==this_stock)]['return']
search10.best_estimator_.predict(this_data10), this_actual10

# TreeExplainer is used to compute SHAP values for tree-based machine learning models for each day. 
explainer1 = shap.TreeExplainer(search1.best_estimator_)
shap_values1 = explainer1.shap_values(this_data1)

explainer2 = shap.TreeExplainer(search2.best_estimator_)
shap_values2 = explainer2.shap_values(this_data2)

explainer3 = shap.TreeExplainer(search3.best_estimator_)
shap_values3 = explainer3.shap_values(this_data3)

explainer4 = shap.TreeExplainer(search4.best_estimator_)
shap_values4 = explainer4.shap_values(this_data4)

explainer5 = shap.TreeExplainer(search5.best_estimator_)
shap_values5 = explainer5.shap_values(this_data5)

explainer6 = shap.TreeExplainer(search6.best_estimator_)
shap_values6 = explainer6.shap_values(this_data6)

explainer7 = shap.TreeExplainer(search7.best_estimator_)
shap_values7 = explainer7.shap_values(this_data7)

explainer8 = shap.TreeExplainer(search8.best_estimator_)
shap_values8 = explainer8.shap_values(this_data8)

explainer9 = shap.TreeExplainer(search9.best_estimator_)
shap_values9 = explainer9.shap_values(this_data9)

explainer10 = shap.TreeExplainer(search10.best_estimator_)
shap_values10 = explainer10.shap_values(this_data10)


# Convert the shap values to list so that we can use to create stacked bar chart
y1 = np.ravel(shap_values1).tolist()
y2 = np.ravel(shap_values2).tolist()
y3 = np.ravel(shap_values3).tolist()
y4 = np.ravel(shap_values4).tolist()
y5 = np.ravel(shap_values5).tolist()
y6 = np.ravel(shap_values6).tolist()
y7 = np.ravel(shap_values7).tolist()
y8 = np.ravel(shap_values8).tolist()
y9 = np.ravel(shap_values9).tolist()
y10 = np.ravel(shap_values10).tolist()


# Plot stacked bar chart of shap values for last 10 days
fig = plt.figure(figsize = (8,10))
plt.bar(feature_list, y1)
plt.bar(feature_list, y2, bottom=y1)
plt.bar(feature_list, y3, bottom=[sum(i) for i in zip(y1, y2)])
plt.bar(feature_list, y4, bottom=[sum(i) for i in zip(y1, y2, y3)])
plt.bar(feature_list, y5, bottom=[sum(i) for i in zip(y1, y2, y3, y4)])
plt.bar(feature_list, y6, bottom=[sum(i) for i in zip(y1, y2, y3, y4, y5)])
plt.bar(feature_list, y7, bottom=[sum(i) for i in zip(y1, y2, y3, y4, y5, y6)])
plt.bar(feature_list, y8, bottom=[sum(i) for i in zip(y1, y2, y3, y4, y5, y6, y7)])
plt.bar(feature_list, y9, bottom=[sum(i) for i in zip(y1, y2, y3, y4, y5, y6, y7, y8)])
plt.bar(feature_list, y10, bottom=[sum(i) for i in zip(y1, y2, y3, y4, y5, y6, y7, y8, y9)])
Days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"]
plt.xlabel("Feature List")
plt.ylabel("Shap Values")
plt.xticks(rotation = 90)
plt.legend(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7", "Day 8", "Day 9", "Day 10"])
plt.title("Stacked bar chart of Shap Values for last 10 days")
plt.show()

st.pyplot(fig)






















