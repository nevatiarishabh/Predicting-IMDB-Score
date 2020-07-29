#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:58:31 2020

@author: n_rishabh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling

dataset = pd.read_csv('movie_metadata.csv')
dataset.head()

#dataset_color = dataset.groupby(by = "color").mean()
#dataset_color 

#dataset.drop('color', axis = 1, inplace = True)
#dataset['color'] = dataset['color'].fillna('color')

X = dataset['facenumber_in_poster']
X = pd.get_dummies(X, columns=['facenumber_in_poster'])





#X = dataset.drop(columns=['imdb_score'])
#X
y = dataset['imdb_score']
y
X = np.array(X)
y = np.array(y)
y = y.reshape(-1,1)
y.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_x.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.ensemble import RandomForestRegressor
RandomForest_model = RandomForestRegressor(n_estimators = 100, max_depth = 10)
RandomForest_model.fit(X_train, y_train)
accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
accuracy_RandomForest