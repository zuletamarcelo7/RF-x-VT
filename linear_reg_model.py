# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:28:48 2023

@author: Marcelo
"""
# this code gives the correlation values of the chosen variables
import pandas as pd

data = pd.read_csv('candidate_var1.csv')
dd = pd.read_csv('export.csv')
df = dd.copy()

# Find columns that are not in the "Variables" column of the data DataFrame
missing_columns = [col for col in df.columns if col not in data["Variables"].values]

# Use DataFrame.loc to select only the columns that are present in the "Variables" column
df = df.loc[:, ~df.columns.isin(missing_columns)]

corr_matrix = df.corr()["target"].abs().sort_values(ascending=False)

print(corr_matrix)

excel_filename = 'corr_matrix_big.xlsx'

corr_matrix.to_excel(excel_filename, index=True)

