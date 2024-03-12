# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:08:18 2024

@author: kdabo
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('export-mini_500.csv', nrows=500)
X, y = data.drop('target', axis=1), data['target']

# Convert categorical columns
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create DMatrix
dtrain_reg = xgb.DMatrix(X_train, y_train)
dtest_reg = xgb.DMatrix(X_test, y_test)

# Train the model
params = {"objective": "reg:squarederror", "tree_method": "hist"}
n = 500
model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=n)

# Extract feature importance
importance = model.get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Print feature importance
print("Feature Importance:")
for feature, importance in sorted_importance:
    print(f"{feature}: {importance}")

# Print model's equation
print("Model Equation:")
for weight in model.get_dump()[0].split("\n"):
    print(weight)
    
# Cross-validation
results = xgb.cv(params, dtrain_reg, num_boost_round=n, nfold=15, early_stopping_rounds=20)

# Extract and print the minimum RMSE
best_rmse = results['test-rmse-mean'].min()
print(f"Best RMSE: {best_rmse}")