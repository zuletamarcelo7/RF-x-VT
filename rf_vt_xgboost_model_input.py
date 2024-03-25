# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:45:08 2024

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

def predict_output(input_values, model):
    """
    Predict output using the trained XGBoost model.
    
    Args:
        input_values (dict): Dictionary containing input values for variables.
        model (xgb.Booster): Trained XGBoost model.
    
    Returns:
        float: Predicted output.
    """
   # Convert user input to DataFrame
    input_df = pd.DataFrame(user_input, index=[0])
    # Convert categorical columns
    input_df = pd.get_dummies(input_df)
    # Get feature importance
    importance = model.get_score(importance_type='weight')
    # Initialize prediction
    prediction = 0
    # Calculate prediction based on feature importance and coefficients
    for feature, importance in importance.items():
        if feature in input_df.columns:
            prediction += input_df[feature].values[0] * model.get_score().get(feature, 0)
    return prediction

# Prompt user to input values for variables
user_input = {}
for column in X.columns:
    value = input(f"Enter value for {column}: ")
    user_input[column] = float(value)

# Predict output based on user input
predicted_output = predict_output(user_input, model)
print("Predicted Output:", predicted_output)