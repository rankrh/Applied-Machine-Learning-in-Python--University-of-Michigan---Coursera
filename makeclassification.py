# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:58:15 2019

@author: Bob
"""

from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.datasets import load_breast_cancer
from matplotlib.colors import ListedColormap
from adspy_shared_utilities import load_crime_dataset
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

plt.figure()
plt.title('Sample Regression -- One Input')
X_R1, y_R1 = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    bias=150.0,
    noise=30,
    random_state=108)

plt.scatter(X_R1, y_R1)


# synthetic dataset for more complex regression
from sklearn.datasets import make_friedman1
plt.figure()
plt.title('Complex regression problem with one input variable')
X_F1, y_F1 = make_friedman1(
    n_samples = 100,
    n_features = 7,
    random_state=0)

plt.scatter(X_F1[:, 2], y_F1, marker= 'o', s=50)
plt.show()

# synthetic dataset for classification (binary) 
plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_C2, y_C2 = make_classification(
    n_samples = 100,
    n_features=2,
    n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y = 0.1,
    class_sep = 0.5, random_state=0)

plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2,
           marker= 'o', s=50, cmap=cmap_bold)
plt.show()


# more difficult synthetic dataset for classification (binary) 
# with classes that are not linearly separable
X_D2, y_D2 = make_blobs(n_samples = 100, n_features = 2, centers = 8,
                       cluster_std = 1.3, random_state = 4)
y_D2 = y_D2 % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2,
           marker= 'o', s=50, cmap=cmap_bold)
plt.show()

