#!/usr/bin/env python

"""
This is am implementation of:
    Romera-Paredes, Bernardino, and P. H. S. Torr.
    "An embarrassingly simple approach to zero-shot learning."
    Proceedings of The 32nd International Conference on Machine Learning. 2015.
"""

from sklearn import datasets, decomposition, preprocessing, manifold
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd
import numpy as np
import collections
import random
import code
import sys

__author__ = "Alexander Elkholy"
__license__ = "MIT"
__copyright__ = "Copyright 2016, Maana Inc."
__maintainer__ = "Alexander Elkholy"
__email__ = "alexanderelkholy@gmail.com"
__status__ = "alpha"

def main():

    runs = 3
    lamb_rng = np.arange(pow(10,-1.1),pow(10,1.1))
    gamma_rng = np.arange(pow(10,-1.1),pow(10,1.1))

    digits_to_try = [4,7,8,3,9,1]

    #The rows in this matrix will represent the lambda parameter;
    #The columns will represent the gamma parameter
    #The lambda and gamma values will increase with the index. That is,
    #The (0,0) entry in the matrix will have the smallest values for
    #lambda and gamma, while the (n,n) (bottom right) entry will have the
    #largest values for lambda/gamma.
    results_mean = np.zeros([np.size(lamb_rng),np.size(gamma_rng)])
    results_std = np.zeros([np.size(lamb_rng),np.size(gamma_rng)])

    for j in lamb_rng:
        for k in gamma_rng:
            runs_correct_percent = []
            for a in range(runs):
                #Process the data
                processed = load_data(digits_to_try)

                #Calculate the matrices on the training data
                S = get_signature(processed.training_data, processed.training_labels) # S is a x z
                try:
                    V = get_V(processed.training_data, processed.training_labels, S, j, k)  #V is d x a
                except np.linalg.linalg.LinAlgError:
                    print "Lambda " + str(j)
                    print "Gamma " + str(k)
                    sys.exit(1)
                W = np.dot(V,S) #W is d x z
                S_prime = get_signature(processed.testing_data, processed.testing_labels)

                #Inference stage
                correct = 0
                #X is already transposed
                for i, (x, y) in enumerate(zip(processed.testing_data.values, np.ravel(processed.testing_labels.values))):
                  if np.argmax(np.dot(x, np.dot(V,S_prime))) == y:
                    correct +=1

                #print "Correct count: " + str(correct)
                #print "Total instances: " + str(i)
                #print "Percentage correct: " + str((correct / float(np.shape(processed.testing_labels.values)[0]))*100)
                runs_correct_percent.append(float(correct) / np.shape(processed.testing_labels.values)[0])

            results_mean[j,k] = np.mean(runs_correct_percent)
            results_std[j,k] = np.std(runs_correct_percent)

    print("Best Average Score: " + str(np.max(results_mean)*100))
    code.interact(local=locals())

def get_V(data, labels, S, lamb, gamma):

    X = np.transpose(data.values) #X is d x m
    Y = pd.get_dummies(np.ravel(labels.values)).values #Y is m x z
    Y[Y==0]=-1

    xxt = np.dot(X,np.transpose(X))
    first_term = np.linalg.inv(xxt + gamma*np.eye(np.shape(xxt)[0]))
    second_term = np.dot(np.dot(X,Y),np.transpose(S))

    sst = np.dot(S,np.transpose(S))
    third_term = np.linalg.inv(sst + lamb*np.eye(np.shape(sst)[0]))

    return np.dot(np.dot(first_term,second_term),third_term)

def get_signature(data, raw_labels):
    """
        Should return a 4 x z* matrix, where z* is the number of classes in the
        labels matrix.
    """

    labels = raw_labels.reset_index()

    pca = decomposition.PCA(n_components=2)
    lle = manifold.LocallyLinearEmbedding(n_components=2)

    X_pca = pd.DataFrame(pca.fit_transform(data))
    X_lle = pd.DataFrame(lle.fit_transform(data))

    class_no = np.shape(labels[0].unique())[0]

    S = np.zeros([4,class_no])
    for a in labels[0].unique():
        this_pca = X_pca.loc[labels.loc[labels[0]==a].index]
        this_lle = X_lle.loc[labels.loc[labels[0]==a].index]
        S[0,a] = this_pca[0].mean()
        S[1,a] = this_pca[1].mean()
        S[2,a] = this_lle[0].mean()
        S[3,a] = this_lle[1].mean()

    return S

def load_data(digits=[]):
    """
        Loads data from sklearn's digits dataset
        (http://scikit-learn.org/stable/datasets/)
        and performs preprocessing.
        ----
        Note that the digits dataset has:
        d = 64   (dimensionality)
        m = ~180 (number of instances per class)
        z = 10   (number of classes)
        digits: An np array which has the digits you want to train on. The
                digits must be in the range of [0,9].
        Output: Returns the train/test, digits and targets data after
                performing preprocessing.
    """

    #Loads the data and the targets, resp.
    #Note they should be indexed the same way. So digits_data[n] corresponds
    #to digits_labels[n] for any n.
    digits_data = pd.DataFrame(datasets.load_digits().data)
    digits_labels = pd.Series(datasets.load_digits().target)

    #If the digits to train on are not specified, pick randomly
    if len(digits) == 0:
        r_digits = range(0,10)
        random.shuffle(r_digits)
        #0-6 is 70% of the data
        training_digits = set()
        testing_digits = set()
        for a in range(0,7):
            training_digits.add(r_digits[a])
        for a in range(7,10):
            testing_digits.add(r_digits[a])
    else:
        if len(digits) > 0:
            #If they specify digits outside of the range, throw
            if (max(digits)>9 or min(digits)<0):
                raise ValueError('The dataset only has digits 0-9. The parameter passed to load_data had a digit outside of that range')
                if len(digits) >= 10:
                    raise ValueError('The dataset only has digits 0-9. You said to train on all of them leaving no testing data')

            all_digits = set([0,1,2,3,4,5,6,7,8,9])
            training_digits = set(digits)
            testing_digits = all_digits - training_digits

    #Training data
    raw_train_labels = digits_labels[digits_labels.isin(training_digits)]
    training_data = digits_data.loc[raw_train_labels.index]
    #Maps the labels to 0...n
    training_labels = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(raw_train_labels))

    #Testing data
    raw_test_labels = digits_labels[digits_labels.isin(testing_digits)]
    testing_data = digits_data.loc[raw_test_labels.index]
    #Maps the labels to 0...n
    testing_labels = pd.DataFrame(preprocessing.LabelEncoder().fit_transform(raw_test_labels))

    processed = collections.namedtuple('processed', ['training_data', 'training_labels', 'testing_data','testing_labels', 'training_digits', 'testing_digits'])
    return processed(training_data,training_labels,testing_data,testing_labels,training_digits,testing_digits)

if __name__ == "__main__":
    main()
