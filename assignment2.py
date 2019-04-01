# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:01:17 2019

@author: Bob
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# PART I: REGRESSION
# =============================================================================

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


def part1_scatter():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);

# =============================================================================
# QUESTION 1
# =============================================================================

""" Write a function that fits a polynomial LinearRegression model on the
training data X_train for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in
sklearn.preprocessing to create the polynomial features and then fit a linear
regression model) For each model, find 100 predicted values over the interval
x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a numpy array. The
first row of this array should correspond to the output from the model trained
on degree 1, the second row degree 3, the third row degree 6, and the fourth
row degree 9."""

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    DEGREES = [1, 3, 6, 9]
    N_POINTS = 100

    X_tr = X_train.reshape(-1, 1)
    
    result = np.zeros([len(DEGREES), N_POINTS])
    predict = np.linspace(0, 10, N_POINTS).reshape(-1, 1)
    
    for i, degree in enumerate(DEGREES):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_tr)
        predict_ = poly.fit_transform(predict)
        
        linreg = LinearRegression()
        predictions = linreg.fit(X_poly, y_train).predict(predict_)
        
        result[i,:] = predictions
        
    return result
        
#print(answer_one())


# =============================================================================
# QUESTION 2
# =============================================================================

"""Write a function that fits a polynomial LinearRegression model on the
training data X_train for degrees 0 through 9. For each model compute the R2 
(coefficient of determination) regression score on the training data as well
as the the test data, and return both of these arrays in a tuple.

This function should return one tuple of numpy arrays (r2_train, r2_test).
Both arrays should have shape (10,)
"""

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    X_tr = X_train.reshape(-1, 1)
    X_tst = X_test.reshape(-1, 1)
    
    train_scores = np.zeros([10])
    test_scores = np.zeros([10])


    for degree in range(10):
        poly = PolynomialFeatures(degree=degree)
        
        X_poly_tr = poly.fit_transform(X_tr)
        X_poly_tst = poly.fit_transform(X_tst)
        
        linreg = LinearRegression()
        
        linreg.fit(X_poly_tr, y_train)
    
        train = r2_score(y_train, linreg.predict(X_poly_tr))
        test = r2_score(y_test, linreg.predict(X_poly_tst))
        
        train_scores[degree] = train
        test_scores[degree] = test
        
    return train_scores, test_scores
        
# =============================================================================
#  QUESRTION THREE
# =============================================================================
def answer_three():
    """
    Based on the R2 scores from question 2 (degree levels 0 through 9), what
    degree level corresponds to a model that is underfitting? What degree
    level corresponds to a model that is overfitting? What choice of degree
    level would provide a model with good generalization performance on this
    dataset?
    
    Hint: Try plotting the R2 scores from question 2 to visualize the
    relationship between degree level and R2. Remember to comment out the
    import matplotlib line before submission.
    
    This function should return one tuple with the degree values in this
    order: (Underfitting, Overfitting, Good_Generalization). There might be
    multiple correct solutions, however, you only need to return one possible
    solution, for example, (1,2,3)"""
    
    fits = answer_two()
        
    bar_width = 0.4
    
    train_scores_x = np.arange(10)
    test_scores_x = [score + bar_width for score in np.arange(10)]
    
    plt.bar(train_scores_x, fits[0], width=bar_width)
    plt.bar(test_scores_x, fits[1], width=bar_width)
    
    plt.xticks(
        [r + bar_width for r in range(10)],
        np.arange(10))
    
    diff = fits[0] - fits[1]
    
    good =  np.where(diff==diff.min())[0][0]
    
    under = np.where(diff==diff[:good].max())[0][0]
    over = np.where(diff==diff[good + 1:].max())[0][0]
        
    return under, good, over
    
    
# =============================================================================
# QUESTION FOUR
# =============================================================================

def question_four():
    """Training models on high degree polynomial features can result in
    overly complex models that overfit, so we often use regularized versions
    of the model to constrain model complexity, as we saw with Ridge and
    Lasso linear regression.
    
    For this question, train two models: a non-regularized LinearRegression
    model (default parameters) and a regularized Lasso Regression model (with
    parameters alpha=0.01, max_iter=10000) both on polynomial features of
    degree 12. Return the R2 score for both the LinearRegression and Lasso
    model's test sets.
    
    This function should return one tuple (LinearRegression_R2_test_score,
    Lasso_R2_test_score)"""
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    
    X_tr = X_train.reshape(-1, 1)
    X_tst =X_test.reshape(-1, 1)
    poly = PolynomialFeatures(degree=12)
    X_poly_tr = poly.fit_transform(X_tr)
    X_poly_tst = poly.fit_transform(X_tst)
        
    linreg = LinearRegression()
    linreg.fit(X_poly_tr, y_train)
    linear = r2_score(y_test, linreg.predict(X_poly_tst))

    lasso = Lasso(alpha=0.01, max_iter=10000)    
    lasso.fit(X_poly_tr, y_train)
    lasso = r2_score(y_test, lasso.predict(X_poly_tst))
    
    
    return linear, lasso

# =============================================================================
# PART II: CLASSIFICATION                                         
# =============================================================================
     
"""Here's an application of machine learning that could save your life! For
this section of the assignment we will be working with the UCI Mushroom Data
Set stored in readonly/mushrooms.csv. The data will be used to train a model
to predict whether or not a mushroom is poisonous. The following attributes
are provided:

Attribute Information:

    cap-shape:
        bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    cap-surface:
        fibrous=f, grooves=g, scaly=y, smooth=s
    cap-color:
        brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u,
        red=e, white=w, yellow=y
    bruises?:
        bruises=t, no=f
    odor:
        almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n,
        pungent=p, spicy=s
    gill-attachment:
        attached=a, descending=d, free=f, notched=n
    gill-spacing: 
        close=c, crowded=w, distant=d
    gill-size: 
        broad=b, narrow=n
    gill-color:
        black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o,
        pink=p, purple=u, red=e, white=w, yellow=y
    stalk-shape:
        enlarging=e, tapering=t
    stalk-root: 
        bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
    stalk-surface-above-ring:
        fibrous=f, scaly=y, silky=k, smooth=s
    stalk-surface-below-ring:
        fibrous=f, scaly=y, silky=k, smooth=s
    stalk-color-above-ring: 
        brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e,
        white=w, yellow=y
    stalk-color-below-ring:
        brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
    veil-type:
        partial=p, universal=u
    veil-color:
        brown=n, orange=o, white=w, yellow=y
    ring-number:
        none=n, one=o, two=t
    ring-type: 
        cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, 
        sheathing=s, zone=z
    spore-print-color: 
        black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u,
        white=w, yellow=y
    population:
        abundant=a, clustered=c, numerous=n, scattered=s, several=v,
        solitary=y
    habitat: 
        grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d


The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables. """
mush_df = pd.read_csv('readonly/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.class_p

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_mush, y_mush,
    random_state=0)

X_subset, y_subset = X_test2, y_test2

# =============================================================================
# QUESTION FIVE
# =============================================================================

def question_five():
    """Using X_train2 and y_train2 from the preceeding cell, train a
    DecisionTreeClassifier with default parameters and random_state=0. What
    are the 5 most important features found by the decision tree?
    
    As a reminder, the feature names are available in the X_train2.columns
    property, and the order of the features in X_train2.columns matches the
    order of the feature importance values in the classifier's
    feature_importances_ property.
    
    This function should return a list of length 5 containing the feature
    names in descending order of importance.
    
    Note: remember that you also need to set random_state in the
    DecisionTreeClassifier."""
    
    from sklearn.tree import DecisionTreeClassifier
    
    tree = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    
    features = pd.Series(
        data=tree.feature_importances_,
        index=X_mush.columns
        ).sort_values(ascending=False
        ).head().index.tolist()
    
    return features


# =============================================================================
# QUESTION SIX
# =============================================================================

def answer_six():
    """
    For this question, we're going to use the validation_curve function in
    sklearn.model_selection to determine training and test scores for a
    Support Vector Classifier (SVC) with varying parameter values. Recall
    that the validation_curve function, in addition to taking an initialized
    unfitted classifier object, takes a dataset as input and does its own
    internal train-test splits to compute results.
    
    Because creating a validation curve requires fitting multiple models, for
    performance reasons this question will use just a subset of the original
    mushroom dataset: please use the variables X_subset and y_subset as
    input to the validation curve function (instead of X_mush and y_mush) to
    reduce computation time.
    
    The initialized unfitted classifier object we'll be using is a Support
    Vector Classifier with radial basis kernel. So your first step is to
    create an SVC object with default parameters (i.e. kernel='rbf', C=1) and
    random_state=0. Recall that the kernel width of the RBF kernel is
    controlled using the gamma parameter.
    
    With this classifier, and the dataset in X_subset, y_subset, explore the
    effect of gamma on classifier accuracy by using the validation_curve
    function to find the training and test scores for 6 values of gamma from
    0.0001 to 10 (i.e. np.logspace(-4,1,6)). Recall that you can specify what
    scoring metric you want validation_curve to use by setting the "scoring"
    parameter. In this case, we want to use "accuracy" as the scoring metric.
    
    For each level of gamma, validation_curve will fit 3 models on different
    subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per
    level) arrays of the scores for the training and test sets.
    
    Find the mean score across the three models for each level of gamma for
    both arrays, creating two arrays of length 6, and return a tuple with the
    two arrays.

    e.g. if one of your array of scores is

        array([[ 0.5,  0.4,  0.6],
               [ 0.7,  0.8,  0.7],
               [ 0.9,  0.8,  0.8],
               [ 0.8,  0.7,  0.8],
               [ 0.7,  0.6,  0.6],
               [ 0.4,  0.6,  0.5]])

    it should then become

        array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])

    This function should return one tuple of numpy arrays (training_scores,
    test_scores) where each array in the tuple has shape (6,).
    """
    
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    clf = SVC(kernel='rbf', C=1.0, random_state=0).fit(X_mush, y_mush)

    train_scores, test_scores = validation_curve(
        clf,
        X_subset, y_subset,
        param_name='gamma',
        param_range=np.logspace(-4,1,6))
    
    return np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)
        
# =============================================================================
# QUESTION 7
# =============================================================================
    
def answer_seven():
    """ Based on the scores from question 6, what gamma value corresponds to a
    model that is underfitting (and has the worst test set accuracy)? What
    gamma value corresponds to a model that is overfitting (and has the worst
    test set accuracy)? What choice of gamma would be the best choice for a 
    model with good generalization performance on this dataset (high accuracy
    on both training and test set)?
    
    Hint: Try plotting the scores from question 6 to visualize the
    relationship between gamma and accuracy. Remember to comment out the
    import matplotlib line before submission.
    
    This function should return one tuple with the degree values in this
    order: (Underfitting, Overfitting, Good_Generalization) Please note
    there is only one correct solution.
    """
    fits = answer_six()
        
    bar_width = 0.4
    
    train_scores_x = np.arange(6)
    test_scores_x = [score + bar_width for score in np.arange(6)]
    
    plt.bar(train_scores_x, fits[0], width=bar_width)
    plt.bar(test_scores_x, fits[1], width=bar_width)
    
    plt.xticks(
        [r + bar_width for r in range(6)],
        np.arange(6))
        
    good = 3
    under = 0
    over = 5
    
    gamma = np.logspace(-4,1,6)
    return gamma[under], gamma[over], gamma[good]

print(answer_seven())

    
    
                                                           
                                                           
                                                           
                                                           