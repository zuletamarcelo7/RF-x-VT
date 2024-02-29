# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:56:43 2024

@author: kdabo
"""

# This is my first attempt at creating the Repubic Finance CLV model using XGBoost
# Still a rough product
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb


warnings.filterwarnings("ignore")
data = pd.read_csv('export.csv', nrows=500)

#print(data.head())
#print(data.shape)

#print(data.describe)

from sklearn.model_selection import train_test_split

X,y = data.drop('target', axis=1), data[['target']]

school = X.select_dtypes(exclude=np.number).columns.tolist()

for col in school:
    X[col] = X[col].astype('category')
#print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

#params = {"objective":"reg:squarederror", "tree_method":"hist"}

#n = 500
#model = xgb.train(
#    params=params,
    #dtrain=dtrain_reg,
    #num_boost_round=n,
#)

#predicted = model.predict(dtest_reg)

# Flattening the arrays
y_test = np.array(y_test).flatten()

#mse = np.mean((y_test - predicted) ** 2)
#rmse = np.sqrt(mse)

#print(f"Mean Squared Error: {mse}")
#print(f"Root Mean Squared Error: {rmse}")

#from sklearn.metrics import mean_squared_error

#preds = model.predict(dtest_reg)
#rmse = mean_squared_error(y_test, preds, squared=False)

#print(f"RMSE of the base mode: {rmse:.3f}")

## Validation

params = {"objective":"reg:squarederror", "tree_method":"hist"}
n = 10000

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]


results = xgb.cv(
    params, dtrain_reg,
    num_boost_round=n,
    nfold=15,
    early_stopping_rounds=20
    )

#print(results.head())

best_rmse = results['test-rmse-mean'].min()

print(best_rmse)