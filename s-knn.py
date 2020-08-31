# -*- coding: utf-8 -*-
#
#  s-knn.py
#
#  Created on 3/27/2020.
#  Authors: Nikil Roashan Selvam, Varun Sivashankar.

import matplotlib as mpl
import matplotlib.pyplot as plt
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sklearn

#Importing the data
def load_data(name):
    if name=='iris':
        iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        df = pd.read_csv(iris, sep=',')
        attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
        df.columns = attributes
        read_in_array = df.to_numpy()

        for i in range(len(read_in_array)):
            if read_in_array[i][4] == 'Iris-setosa':
                read_in_array[i][4] = np.int64(0)
            elif read_in_array[i][4] == 'Iris-versicolor':
                read_in_array[i][4] = np.int64(1)
            elif read_in_array[i][4] == 'Iris-virginica':
                read_in_array[i][4] = np.int64(2)
            else:
                print("error")
        return read_in_array


    elif name=='balance-scale':
        df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data')
        read_in_array = df.to_numpy()
        one_hot = np.zeros((len(read_in_array),3))
        for i in range(len(read_in_array)):
            if read_in_array[i][0] == 'L':
                one_hot[i][0] = 1
            elif read_in_array[i][0] == 'B':
                one_hot[i][1] = 1
            elif read_in_array[i][0] == 'R':
                one_hot[i][2] = 1    
                read_in_array = np.append(one_hot,read_in_array[:,1:],axis=1)
        return read_in_array


    elif name=='yeast':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data', delim_whitespace=True)
        read_in_array = df.to_numpy()
        read_in_array=read_in_array[:,1:]
        yeast_labels={}
        for idx,yeast_class in enumerate(read_in_array[:,-1]):
            if yeast_class not in yeast_labels:
                yeast_labels[yeast_class]=np.int64(len(yeast_labels))
            read_in_array[idx][-1]=yeast_labels[yeast_class]
        return read_in_array

    elif name=='blood':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
        read_in_array = df.to_numpy()
        return read_in_array

    elif name=='haberman':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data')
        read_in_array = df.to_numpy()
        return read_in_array

    elif name=='ion':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data')
        read_in_array = df.to_numpy()
        for i in range(len(read_in_array)):
            if read_in_array[i][-1] == 'b':
                read_in_array[i][-1] = 0
            elif read_in_array[i][-1] == 'g':
                read_in_array[i][-1] = 1
        return read_in_array
    
    elif name=='red-wine':
      df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
      read_in_array = df.to_numpy()
      return read_in_array

    else:
        raise ValueError("Invalid dataset name: ", name)




def scaledKNN(data,num_epochs=1, alpha_0=1e-6, start_from_ones=False):
    data_copy = np.copy(data)
    n = len(data_copy[0])-1

    #create dictionary of class against indices
    class_to_inds = {}
    for i in range(len(data_copy)):
        c = data_copy[i][-1]
        if c in class_to_inds:
            class_to_inds[c].append(i)
        else:
            class_to_inds[c] = [i]

    class_to_dists = {}
    dists = np.zeros((1,n))
    for c in class_to_inds:
        temp = np.zeros((1,n))
        inds = class_to_inds[c]
        for i in range(len(inds)):
            point1 = data_copy[i][:-1]
            for j in range(i+1,len(inds)):
                point2 = data_copy[j][:-1]
                temp = temp + (point1-point2)**2
        class_to_dists[c] = temp
        dists = dists + temp
    dists = dists.T
    
    #initialize weights from uniform distribution between 0 and 1
    if start_from_ones:
        weights = np.ones((n,1))
    else:
        weights = np.random.rand(n,1)

    cost = 0
    old_cost = None
    grad = np.zeros((n,1))
    beta = 0.9 # momentum parameter
    alpha = alpha_0
    num_epoch = 0
    
    while num_epoch < num_epochs:
        cost = np.sum(np.multiply(weights**2,dists)) + np.sum(1/weights)
        new_grad = 2 * np.multiply(weights,dists) - 1/weights**2
        grad=(1-beta)*grad + beta*new_grad
        weights = weights - alpha * grad

        for i in range(len(weights)):
            if weights[i] < 0: weights[i] = -weights[i]
        weights = n * weights/np.sum(weights)

        if old_cost != None and old_cost < cost:
            alpha = alpha_0/(1+20*num_epoch)
            
        old_cost = cost
        num_epoch += 1
        
    return weights, cost


# MAIN SCRIPT

datasets = ["iris", "balance-scale", "yeast", "blood", "haberman","ion","red-wine"]
seeds = [1,12,123,1234,12345,123456,1234567,12345678,123456789]
alphas = {'iris': 1e-08, 'balance-scale': 1e-05, 'yeast': 1e-07, 'blood': 1e-08, 'haberman': 0.001, 'ion': 1e-08, 'red-wine': 1e-06}
num_epochs = 6000

for dataset in datasets:
    print("\t\t--Dataset: ", dataset)
    data = load_data(dataset)
    data = np.array(data, dtype=np.float64)
    
    for seed in seeds:
        print("\t\t--Seed: ", seed)
        
        #create train-test split
        np.random.seed(seed)
        np.random.shuffle(data)
        split = int(0.8*len(data))
        train, test = data[:split,:], data[split:,:]

        # normalize data
        stds = np.std(train, axis = 0)
        means = np.mean(train, axis = 0)
        n = len(data[0]) - 1
        epsilon = 10**(-8)

        for i in range(n):
            stds[i] = (stds[i]**2 + epsilon)**0.5

        for i in range(n):
            train[:,i] = (train[:,i] - means[i]) / stds[i]
            test[:,i] = (test[:,i] - means[i]) / stds[i]

        alpha = alphas[dataset]

        weights,_ = scaledKNN(train,num_epochs=num_epochs,alpha_0=alpha,start_from_ones=True)
        weights = np.array(weights).T * np.identity(len(weights))

        #Get Results
        train_X = train[:,0:-1]
        train_y = list(train[:,-1])
        test_X = test[:,0:-1]
        test_y = list(test[:,-1])

        cur_results = []

        #Vanilla kNN
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(train_X,train_y)

        train_ypred = clf.predict(train_X)
        acc = metrics.accuracy_score(train_y, train_ypred, normalize=True)
        print('Vanilla kNN:\t-- train acc %.3f' % acc)

        test_ypred = clf.predict(test_X)
        acc = metrics.accuracy_score(test_y, test_ypred, normalize=True)
        print('Vanilla kNN:\t-- test acc %.3f' % acc)

        #Scaled kNN
        clf = KNeighborsClassifier(n_neighbors=5)
        train_X = np.dot(train_X, weights)
        test_X = np.dot(test_X, weights)
        clf.fit(train_X,train_y)

        train_ypred = clf.predict(train_X)
        acc = metrics.accuracy_score(train_y, train_ypred, normalize=True)
        print('Scaled kNN:\t-- train acc %.3f' % acc)

        test_ypred = clf.predict(test_X)
        acc = metrics.accuracy_score(test_y, test_ypred, normalize=True)
        print('Scaled kNN:\t-- test acc %.3f' % acc)
