# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:05:47 2019

@author: Bob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split


fruits = pd.read_table('fruit_data_with_colors.txt')

print(fruits.describe())

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(
    fruits,
    c=fruits.fruit_label,
    marker='.',
    hist_kwds={'bins': 15},
    figsize=(12,12),
    cmap=cmap)