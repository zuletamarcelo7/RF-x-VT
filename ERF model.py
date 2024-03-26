# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:06:50 2024

@author: tweiq
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
csv_file = 'red_data_250.csv'
df = pd.read_csv(csv_file)

data = pd.read_csv("red_data-mini_250.csv") ##random customers you input for prediction
customers = data.drop(columns=['target'])

# Data preprocessing
X = df.drop(columns=['target'])
X = SimpleImputer(strategy='mean').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)
y = df['target']

alpha_values = [0.1, 0.5, 1, 2]

mse_results = []

for alpha in alpha_values:
    ridge_reg = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_reg, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    mse_results.append({'Alpha': alpha, 'MSE': mse})

mse_df = pd.DataFrame(mse_results)
ranked_mse = mse_df.sort_values(by='MSE')

print("Ranked MSE values (Lowest to Highest):")
print(ranked_mse)