# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:14:06 2023

@author: Marcelo
"""
# This piece of code shows the correlation of a given variable with the target variable
# The code will output a scatter plot showing the available data points
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('export-mini_500.csv')

# Specify the variable names
target_var = 'target'
var_name = 'addrlastmoveecontrajectoryindex' # input test variable name here

# Extract the values from the DataFrame
target_values = data[target_var]
var1_values = data[var_name]

# Create a scatter plot
plt.scatter(target_values, var1_values)
plt.title(f'Scatter Plot of {target_var} vs {var_name}')
plt.xlabel(f'{target_var} Values')
plt.ylabel(f'{var_name} Values')
plt.show()

#ignore 
#for item in var1_values:
#    print(item)