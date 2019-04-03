# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:33:12 2019

@author: Bob
"""

import numpy as np
import pandas as pd

# =============================================================================
# QUESTION ONE
# =============================================================================
"""Import the data from fraud_data.csv. What percentage of the observations in
the dataset are instances of fraud? This function should return a float
between 0 and 1.
"""

def answer_one():
    fraud = pd.read_csv('readonly/fraud_data.csv')
    
    fraud_rate = len(fraud[fraud.Class == 1]) / len(fraud)
    
    return fraud_rate


# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('readonly/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# =============================================================================
# QUESTION TWO
# =============================================================================
"""Using X_train, X_test, y_train, and y_test (as defined above), train a
dummy classifier that classifies everything as the majority class of the
training data. What is the accuracy of this classifier? What is the recall?

This function should a return a tuple with two floats, i.e. (accuracy score,
recall score)."""

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    
    dummy_majority = DummyClassifier(
        strategy="most_frequent"
        ).fit(
            X_train, y_train)
    
    accuracy = dummy_majority.score(X_test, y_test)
    recall = recall_score(
            y_test,
            dummy_majority.predict(X_test))
    
    return accuracy, recall


# =============================================================================
# QUESTION THREE
# =============================================================================

"""Using X_train, X_test, y_train, y_test (as defined above), train a SVC
classifer using the default parameters. What is the accuracy, recall, and
precision of this classifier?  This function should a return a tuple with
three floats, i.e. (accuracy score, recall score, precision score)"""

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    
    fraud_svc = SVC().fit(X_train, y_train)
    accuracy = fraud_svc.score(X_test, y_test)
    
    fraud_svc_predictions = fraud_svc.predict(X_test)
    
    recall = recall_score(y_test, fraud_svc_predictions)
    precision = precision_score(y_test, fraud_svc_predictions)
    
    return accuracy, recall, precision

# =============================================================================
# Question 4
# =============================================================================

"""Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what
is the confusion matrix when using a threshold of -220 on the decision
function. Use X_test and y_test.  This function should return a confusion
matrix, a 2x2 numpy array with 4 integers."""


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    fraud_decision = SVC(
            C=1e9,
            gamma=1e-7
        ).fit(
            X_train,
            y_train
        ).decision_function(X_test)
    
    fraud_decision = np.where(fraud_decision >= -220, 1, 0)
    
    confusion = confusion_matrix(y_test, fraud_decision)
    return confusion

# =============================================================================
# Question 5
# =============================================================================

"""Train a logisitic regression classifier with default parameters using
X_train and y_train.  For the logisitic regression classifier, create a
precision recall curve and a roc curve using y_test and the probability
estimates for X_test (probability it is fraud).  Looking at the precision
recall curve, what is the recall when the precision is 0.75?  Looking at the
roc curve, what is the true positive rate when the false positive rate is
0.16?  This function should return a tuple with two floats, i.e. (recall,
true positive rate).
"""

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc
    
    logistic = LogisticRegression().fit(X_train, y_train)
    decision = logistic.decision_function(X_test)
    
    precision, recall, threshold = precision_recall_curve(y_test, decision)
    
    precision75 = np.where(precision==0.75)
    recall75 = float(recall[precision75])
    
    false_positive, true_positive, thresholds = roc_curve(y_test, decision)
    
    false_positive16 = np.where(np.round(false_positive, 2)==0.16)
    true_positive16 = true_positive[false_positive16][0]
    
    return recall75, true_positive16


# =============================================================================
# Question 6
# =============================================================================

"""Perform a grid search over the parameters listed below for a Logisitic
Regression classifier, using recall for scoring and the default 3-fold cross
validation.

    'penalty': ['l1', 'l2']
    'C':[0.01, 0.1, 1, 10, 100]

From .cv_results_, create an array of the mean test scores of each parameter
combination.This function should return a 5 by 2 numpy array with 10 floats.
"""

def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    logistic = LogisticRegression()
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C':[0.01, 0.1, 1, 10, 100]}
    
    grid = GridSearchCV(logistic, param_grid=param_grid)
    
    return grid
        
    
print(answer_six())



