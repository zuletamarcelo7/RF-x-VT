# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:00:54 2023

@author: Marcelo
"""
# This code calculate the correlation of a single variable across the entire 
# population of customers. Specify variable name accordingly and correlation matrix
# will appear in the console window

import pandas as pd

data = pd.read_csv('export.csv', nrows=93986)


target = []
var1 = []
var_name = "CRP02_0903018000" #specify variable name here

for item in data["target"]:
    target.append(item)
for item in data[var_name]:
    var1.append(item)
    
print(target)
print(var1)

book = {
        "target": target,
        var_name: var1
        }

dataframe = pd.DataFrame(book, columns=["target", var_name])
print(dataframe)

matrix = dataframe.corr()
print("Correlation matrix is : ")
print(matrix)
