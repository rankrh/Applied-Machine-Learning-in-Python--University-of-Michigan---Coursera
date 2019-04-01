# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:25:53 2019

@author: Bob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('fruit_data_with_colors.txt')

lookup_fruit_name = dict(zip(
    fruits.fruit_label.unique(),
    fruits.fruit_name.unique()))

X = fruits[['mass', 'width', 'height', 'color_score']]
Y = fruits.fruit_label

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=108)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_train)

print(knn.score(X_test, Y_test))

print(knn.predict([[20, 4.3, 5.5]])

