# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:48:22 2023

@author: Marcelo
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def generate_regression_equation(model, feature_names):
    intercept = model.intercept_
    coefficients = model.coef_
    
    equation = f"Y = {intercept:.4f}"
    
    for feature, coefficient in zip(feature_names, coefficients):
        equation += f" + {coefficient:.4f} * {feature}"
    
    return equation

df = pd.read_csv("reduced2.csv")
data = pd.read_csv("sample_customers2.csv") # this is where we would input potential customers
customers = data.drop(columns=['Unnamed: 0'])

# Selecting independent variables (X) and dependent variable (y)
X = df.drop(columns=['target'])
y = df['target']

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose 'median', 'most_frequent', etc.
X_imputed = imputer.fit_transform(X)

# Fit the model
model = LinearRegression()
model.fit(X_imputed, y)

# Get feature names
feature_names = X.columns

# Generate the regression equation
equation = generate_regression_equation(model, feature_names)

print("Regression Equation:")
print(equation)

# Handle null values in the new samples
new_df_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)

# Predict the target variable for new samples
predictions_with_nulls = model.predict(new_df_imputed)
print()
print("Predictions with null values handled:", predictions_with_nulls)


for item in predictions_with_nulls:
    print(item)
    
count = 0
num_cust = 93985
total = 0
while count < num_cust:
    total = total + abs(predictions_with_nulls[count] - df['target'][count])
    count = count + 1
print(total / num_cust)
