# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:54:36 2023

@author: Marcelo
"""

import pandas as pd

data = pd.read_csv('export-mini_2500_99.csv')
df = data.copy()
dd = data.copy()
cust_size = 2500

#corrs = {}
for col in df.columns:
    if df[col].notnull().sum() / cust_size < 0.75:
        del df[col]
    elif df[col].std() == 0:
        del df[col]
    else:
        count = 0
        for item in df[col]:
            if item > 9990 and item <10001:
                count = count + 1
            elif item < 1000000010 and item > 999999990:
                count = count + 1
            elif item == -1:
                count = count + 1
        if count / cust_size > 0.25:
            del df[col]
    
corr_matrix = df.corr()["target"].abs().sort_values(ascending=False)

print(corr_matrix)

excel_filename = 'corr_matrix_2500_test5.xlsx'

corr_matrix.to_excel(excel_filename, index=True)