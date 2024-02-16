# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:57:41 2024

@author: tweiq
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# Load your data
excel_file = 'export-mini_500.xlsx'  # Update with your Excel file name
df = pd.read_excel(excel_file, sheet_name='export-mini_500', engine='openpyxl', index_col=0)  # Assuming the first column is an index

# Extract features (X) and target variable (y)
X = df.iloc[:, 1:]  # Exclude the first column (customer numbers) from features
y = df.index  # Use the index as the target variable (assuming it contains meaningful values)

# Create a DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Evaluate the model using cross-validation
scores = cross_val_score(regressor, X, y, cv=3, scoring='neg_mean_squared_error')

# Print the mean squared error
print('Mean Squared Error:', -scores.mean())