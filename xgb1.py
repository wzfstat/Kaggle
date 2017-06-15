# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 01:06:31 2017

@author: Zhifeng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import os

#os.chdir(r'D:\Dropbox\Kaggle\Sberbank')
# DATA_DIR = ''
DATA_DIR = 'data/Sberbank/'

train = pd.read_csv(DATA_DIR + 'train.csv', parse_dates=['timestamp'], index_col='id')
test = pd.read_csv(DATA_DIR + 'test.csv', parse_dates=['timestamp'], index_col='id')
#macro = pd.read_csv('macro.csv')
fx = pd.read_excel(DATA_DIR + 'BAD_ADDRESS_FIX.xlsx').drop_duplicates('id').set_index('id')
train.update(fx)
test.update(fx)
print('Fix in train: ', train.index.intersection(fx.index).shape[0])
print('Fix in test : ', test.index.intersection(fx.index).shape[0])

id_test = test.index.values

y_train = train["price_doc"]
x_train = train.drop(["price_doc"], axis=1)

df_all = pd.concat([x_train, test])

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_all[c].values)) 
        df_all[c] = lbl.transform(list(df_all[c].values))
        #df_all.drop(c,axis=1,inplace=True)

df_all.loc[df_all.build_year == 20052009, 'build_year'] = 2005
df_all.loc[df_all.build_year == 0, 'build_year'] = np.nan
df_all.loc[df_all.build_year == 1, 'build_year'] = np.nan
df_all.loc[df_all.build_year == 20, 'build_year'] = 2000
df_all.loc[df_all.build_year == 215, 'build_year'] = 2015
df_all.loc[df_all.build_year == 3, 'build_year'] = np.nan
df_all.loc[df_all.build_year == 4965, 'build_year'] = 1965
df_all.loc[df_all.build_year == 71, 'build_year'] = 1971


# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)

df_all['year_old'] = 2020 - df_all.build_year
df_all['floor_inverse'] = df_all.max_floor - df_all.floor
df_all['non_res_sq'] = df_all.full_sq - df_all.life_sq
df_all['per_room_sq'] = df_all.life_sq / df_all.num_room
df_all.loc[df_all.state == 33, 'state'] = 3
df_all.drop(["ID_metro","ID_railroad_station_walk","ID_railroad_station_avto",
            "ID_big_road1","ID_big_road2","ID_railroad_terminal","ID_bus_terminal"], axis=1, inplace=True)

trn_X = df_all.iloc[:x_train.shape[0],:]
tst_X = df_all.iloc[x_train.shape[0]:,:]
trn_y = y_train.values



xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(trn_X, trn_y)
dtest = xgb.DMatrix(tst_X)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=50,
    verbose_eval=20, show_stdv=False)
#cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


num_boost_rounds = np.argmin(cv_output['test-rmse-mean'])
model = xgb.train(xgb_params, dtrain, num_boost_rounds, [ (dtrain,'train') ], verbose_eval=50)

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)


y_pred = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

output.to_csv('xgbsub1.csv', index=False)