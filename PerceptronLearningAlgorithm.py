# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:20:37 2017

@author: brandonjamesmcmahan
Perceptron Learning Algorithm 
"""
import numpy as np
import random
import matplotlib.pyplot as plt
#import time

#create target function
#generate two random points in X
x1 = random.uniform(-1,1) 
y1 = random.uniform(-1,1)

x2 = random.uniform(-1,1) 
y2 = random.uniform(-1,1)

m = (y2 - y1) / (x2 - x1)
b = y1 - m*x1

#definition of target function
def targetFunction(m,b,x1,x2):
    if x2 > m*x1 + b:
        return +1
    elif x2 < m*x1 + b:
        return -1
#end creation of target function
        
#begin experiment, we will iterate 1000 times
#counts = np.zeros(2)
#error = np.zeros(2)
#for iteration in range(0, 1):
#reinsert all code here
N = 100;
#generate training data
x = np.zeros((3,N))
y = np.zeros(N)
for n in range(0,N):
    x[0][n] = 1     #we need this for PLA
    x[1][n] = random.uniform(-1,1)
    x[2][n] = random.uniform(-1,1)
    y[n] = targetFunction(m,b,x[1][n],x[2][n])


x_series = np.linspace(-1,1,1000)
#extract negative and positive points
neg_pts_x1 = []
neg_pts_x2 = []
pos_pts_x1 = []
pos_pts_x2 = []
for i in range(0,N):
    if y[i] < 0:
        neg_pts_x1.append(x[1][i])
        neg_pts_x2.append(x[2][i])    
    elif y[i] > 0:
        pos_pts_x1.append(x[1][i])
        pos_pts_x2.append(x[2][i])

#initialize weight matrix
w = np.zeros(3)
counts = 0
#loop over iterations to get hypothesis set to converge
while not np.all(np.sign(np.dot(w,x)) == y):
    for n in range(0,N):
        if y[n]!=np.sign(np.dot(w,[x[0][n], x[1][n], x[2][n]])):
            w = w + [y[n]*x[0][n], y[n]*x[1][n], y[n]*x[2][n]]
        else:
            w = w
        counts = counts +1
        print "In iteration %.2d" %counts
        #plot random training data
#
#        time.sleep(5)
#        query = 'Press enter to continue to iteration %.2d' %counts
#        raw_input(query)

#determine how wel algorithm does
#generate a very large test space
#test_space = 10**5
#test_data = np.zeros((2,test_space))
#test_results = np.zeros(test_space)
#for n in range(0,test_space):
#    test_data[0][n] = random.uniform(-1,1)
#    test_data[1][n] = random.uniform(-1,1)
#    test_results[n] = targetFunction(m,b,test_data[0][n],test_data[1][n])
#        
#no_misclassified = 0.0    
#for i in range(0,test_space):
#    if (np.sign(np.dot(w,[test_data[0][i],test_data[1][i]])) != test_results[i]):
#            no_misclassified = no_misclassified + 1
#                
#error = 100*no_misclassified/(test_space)

error = 100.0
#count = np.mean(counts)
#err = np.mean(error)
print "\n \nPerceptron Learning Algorithm Results"
print"%f iterations were required for convergence" %counts
print "Classifiction is %.2f%% accurate" %error
print"\n\n"



plt.scatter(neg_pts_x1,neg_pts_x2,c='b')
plt.scatter(pos_pts_x1,pos_pts_x2,c='r')
plt.plot(x_series, -(x_series*w[1]+w[0])/w[2])
plt.title("Perceptron Learning Algorithm")
plt.xlabel("X1")
plt.ylabel("X2")