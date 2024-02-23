# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:57:41 2024

@author: tweiq
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
# Load your data
excel_file = 'test 1.xlsx'  # Update with your Excel file name
df = pd.read_excel(excel_file, sheet_name='Sheet1', engine='openpyxl', index_col=0)  

# Extract features (X)
X = df.iloc[:, :]  # Exclude the first column (customer numbers) from features

# Iterate through each row as the target variable
for index, row in df.iterrows():
    y = row.iloc[:].values  # Use the values in the row as the target variable, excluding the first element

    # Create a DecisionTreeRegressor
    regressor = DecisionTreeRegressor(max_depth=5)
    regressor.fit(X,y)
    
    # Fit an Isolation Forest model to identify outliers
   

    try:
        # Evaluate the model using cross-validation
        scores = cross_val_score(regressor, X, y, cv=3, scoring='neg_mean_squared_error')

        # Print the mean squared error
        mse = -scores.mean()
        print(f'Mean Squared Error for row {index}: {mse}')

        # Check if the MSE is below the threshold
        if mse < 30:
            print(f'MSE for row {index} is below 30.')
        else:
            print(f'MSE for row {index} is above or equal to 30.')
        print('---')

    except Exception as e:
        # Print the exception if an error occurs during model training
        print(f'Error for row {index}: {e}')