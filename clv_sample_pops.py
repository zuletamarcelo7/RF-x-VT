# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:48:42 2023

@author: Marcelo
"""
# This code will assist in creating sample populations. You just need to specify
# size of sample population and seed value. Note: seed value is used to generate randomness

import pandas as pd

df_original = pd.read_csv('export.csv', nrows=93986)

seed_value = 99 # specify seed value
df_sample = df_original.sample(n=2500, random_state=seed_value) # specify sample size

csv_file_path = 'export-mini_2500_99.csv' # specify name of your output csv file
df_sample.to_csv(csv_file_path, index=False)
