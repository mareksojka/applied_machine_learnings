# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:55:36 2017

@author: marek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
X_train=X_train.reshape((11,1))
X_test=X_test.reshape((4,1))
y_train=y_train.reshape((11,1))
y_test=y_test.reshape((4,1))

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
   # %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
# part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    X_predict=np.linspace(0,10,100)
    Predicted_values=np.empty([4,100])
    i=0

    for n in [1,3,6,9]:
        poly = PolynomialFeatures(degree=n)
        X_poly_train = poly.fit_transform(X_train.reshape((11,1)))
        col,row = X_poly_train.shape
        linreg = LinearRegression().fit(X_poly_train.reshape((col,row)), y_train.reshape((11,1)))
        X_predict_poly = poly.fit_transform(X_predict.reshape((100,1)))
        Y_predict = linreg.predict(X_predict_poly)
        Predicted_values[i,:] = Y_predict.reshape((1,100))
        i+=1
    
    return Predicted_values


# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

#plot_one(answer_one())

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    i=0
    r2_train=np.zeros(10,)
    r2_test=np.zeros(10,)
    for n in range(10):
        poly = PolynomialFeatures(degree=n)
        X_poly_train = poly.fit_transform(X_train)
        col,row = X_poly_train.shape
        linreg = LinearRegression().fit(X_poly_train.reshape((col,row)), y_train)
        r2_train[i]=linreg.score(X_poly_train.reshape((col,row)),y_train)
        X_poly_test = poly.fit_transform(X_test)
        col2,row2 = X_poly_test.shape
        r2_test[i]=linreg.score(X_poly_test.reshape((col2,row2)),y_test)
        i+=1

    return (r2_train,r2_test)

def answer_three():
    
    # Your code here
    (r2_train,r2_test)=answer_two()
    Underfitting=np.argmin(r2_train)
    Overfitting=np.argmax(r2_train-r2_test)
    Good_Generalization=np.argmax(r2_test)
    
    return (Underfitting, Overfitting, Good_Generalization)




