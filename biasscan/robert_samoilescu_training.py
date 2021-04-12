#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder

from BiasScan.src.MDSS import *
from BiasScan.src.score import *
from BiasScan.src.priority import *

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'notebook')

from collections import OrderedDict
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from itertools import count
import time


# In[ ]:


NUM_CPUS = cpu_count() 
NUM_EXPERIMENTS = 100


# ### Load Dataset

# In[ ]:


covariates = ["age", "workclass", "education", "marital-status", "occupation", 
              "race", "sex", "hours-per-week"]
outcome = "outcome"

# data containing categorical features
mdscan_data_train = pd.read_csv("datasets/adult/train.data")
mdscan_data_test = pd.read_csv("datasets/adult/test.data")
mdscan_data_both = pd.concat([mdscan_data_train, mdscan_data_test], axis=0, ignore_index=True)

# transform outcome into binary
mdscan_data_train["outcome"] = (mdscan_data_train["outcome"] == " >50K").astype(np.int)
mdscan_data_test["outcome"] = (mdscan_data_test["outcome"] == " >50K.").astype(np.int)


# ### Define Classifiers

# In[ ]:


classifiers_labels = [
    "LogisticRegression",
    "RandomForestClassifier",
    "SVC",
    "KNeighborsClassifier",
    "MLPClassifier"
]


# In[ ]:


domains = {}

for cov in covariates:
    domains[cov] = np.unique(mdscan_data_both[cov])
    
observed = 'outcome'
expected = 'proba' 


# ### Utils

# In[ ]:


def model_selection(X: np.array, y: np.array, clf_label: str):
    """
    Returns the best model using cross validation on the dataset passed as argument
    @param X: feature matrix
    @param y: outcome matrix
    @param clf_label: classifier label [LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier]
    :return: best classifier
    """
    
    if clf_label == "LogisticRegression":
        return LogisticRegression(solver='lbfgs', max_iter=10000)
    
    if clf_label == "RandomForestClassifier":
        randforest = RandomForestClassifier()
        parameters = {
            'n_estimators': [100, 200, 300, 400],
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 8, 16, None]
        }
        clf = GridSearchCV(randforest, parameters, cv=5, scoring='roc_auc', iid=True, n_jobs=1)
        clf = clf.fit(X, y)
        return clf.best_estimator_
    
    if clf_label == "SVC":
        svc = SVC(kernel='rbf', gamma='auto', probability=True)
        return svc
    
    if clf_label == "KNeighborsClassifier":
        knn = KNeighborsClassifier()
        parameters = {
            'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        }
        clf = GridSearchCV(knn, parameters, cv=5, scoring='roc_auc', iid=True, n_jobs=1)
        clf = clf.fit(X, y)
        return clf.best_estimator_
    
    # if clf_label == "MLPClassifier"
    mlp = MLPClassifier(hidden_layer_sizes=64, max_iter=10000)
    return mlp


# # ### Test 1

# # We drastically reduce the available training data (uniformly).  This is where we train logistic regression but only on 1% of the data.  This model will have low AUC -- but the question is is the model making systematic errors against a particular subpopulation.  This is what is addressed by bias scan.  We will then see how AUC and bias changes when scanning over 2% of data.  Then 4%, 8%, 16% and 32% of the available data.  As training data increases we expect to see less bias -- but it isn't clear how different models will respond to this.

# # In[ ]:


# def experiment1(p: int, i: int, seed: int):
#     """
#     @param p: percentage of the population to be kept
#     @param i: experiment index
#     @param seed: experiment seed
#     """
#     np.random.seed(seed)
    
#     results = {}
#     num_rows = mdscan_data_train.shape[0]
    
#     # select with replacement p% of the total number of rows
#     indexes = np.random.choice(np.arange(num_rows), size=num_rows * p // 100, replace=True)
#     sub_train_data = train_data.loc[indexes, :].to_numpy()

#     # first column is the outcome
#     X_train, y_train = sub_train_data[:, 1:], sub_train_data[:, 0]
#     X = test_data.to_numpy()[:, 1:]

#     for clf_label in classifiers_labels:
#         # train the classifier on smaller dataset
#         clf = model_selection(X=X_train, y=y_train, clf_label=clf_label)
#         clf = clf.fit(X_train, y_train)

#         # predict the probability on the entier datast
#         proba = clf.predict_proba(X)[:, 1]
        
#         # save probability in a file
#         filename = "./training/test1/clf=%s_p=%d_i=%d.pkl" % (clf_label, p, i)
#         file = open(filename, 'wb')
#         pkl.dump(proba, file)
#         file.close()


# # In[ ]:


# percentages = [1, 2, 4, 8, 16, 32]
# num_experiments = NUM_EXPERIMENTS

# # transform categorical variables into dummy
# train_data = pd.get_dummies(mdscan_data_train, prefix=covariates, columns=covariates)
# test_data = pd.get_dummies(mdscan_data_test, prefix=covariates, columns=covariates)

# # start time
# start = time.time()

# # define thread pool
# pool = Pool(processes=NUM_CPUS)
    
# for p in percentages:
#     # get a seed for each thread. Otherwise they will gave the same result
#     seeds = np.random.randint(0, 99999, size=num_experiments)
    
#     for i in range(num_experiments):
#         pool.apply_async(experiment1, (p, i, seeds[i]))
        
# # close thread pool & wait for all jobs to be done
# pool.close()
# pool.join()

# # print duration
# print("Ellapsed: %.2f" % (time.time() - start))


# # ### Test 2

# # We remove some features from the training data. Note we remove some rows as well (50%). We first remove 1 feature at random. Then 2 features at random. Then 3, 4, 5 features at random. This allows us to to bootstrap samples and train a half dozen different models that all have race removed but each of the models is trained on a different sample of rows that are sampled at 50% with replacement.

# # In[ ]:


# def experiment2(nc, i, seed):
#     """
#     @param nc: number of covariates/features to be removed
#     @param i: experiment index
#     @param seed: experiment seed
#     """
#     np.random.seed(seed)
    
#     results = {}
#     mdscan_data_train_ = mdscan_data_train.copy()
#     mdscan_data_test_ = mdscan_data_test.copy()
#     num_rows = mdscan_data_train_.shape[0]
    
#     # select at random covariates to be dropped
#     covariates_dropped = np.random.choice(covariates, size=nc, replace=False)

#     # drop covariates from dataframe
#     sub_mdscan_data_train = mdscan_data_train_.drop(covariates_dropped, axis=1)
#     sub_mdscan_data_test = mdscan_data_test_.drop(covariates_dropped, axis=1)

#     # convert sub_mdscan_data to one hot encodings / dummy variables
#     columns = list(set(covariates).difference(set(covariates_dropped)))
#     train_data = pd.get_dummies(sub_mdscan_data_train, prefix=columns, columns=columns)
#     test_data = pd.get_dummies(sub_mdscan_data_test, prefix=columns, columns=columns)
    
#     # select rows for training
#     indexes = np.random.choice(np.arange(num_rows), size=num_rows // 2, replace=True)
#     sub_train_data = train_data.loc[indexes, :].to_numpy()

#     # first column is the outcome
#     X_train, y_train = sub_train_data[:, 1:], sub_train_data[:, 0]
#     X = test_data.to_numpy()[:, 1:]

#     for clf_label in classifiers_labels:
#         # train the classifier on smaller dataset
#         clf = model_selection(X=X_train, y=y_train, clf_label=clf_label)
#         clf = clf.fit(X_train, y_train)

#         # predict the probability on the entier datast
#         proba = clf.predict_proba(X)[:, 1]

#         # save probability in a file
#         filename = "./training/test2/clf=%s_nc=%d_i=%d.pkl" % (clf_label, nc, i)
#         file = open(filename, 'wb')
#         pkl.dump(proba, file)
#         file.close()


# # In[ ]:


# num_covariates = [1, 2, 3, 4, 5]
# num_experiments = NUM_EXPERIMENTS

# # start time
# start = time.time()

# # define thread pool
# pool = Pool(processes=NUM_CPUS)

# for nc in num_covariates:
#     seeds = np.random.randint(0, 99999, size=num_experiments)
#     for i in range(num_experiments):
#         pool.apply_async(experiment2, (nc, i, seeds[i]))
        
# # close thread pool & wait for all jobs to be done
# pool.close()
# pool.join()

# # print duration
# print("Ellapsed: %.2f" % (time.time() - start))


# # ### Utils

# # In[ ]:


# def random_subpopulation(df_train: pd.DataFrame, df_test: pd.DataFrame, domains: OrderedDict, num_cov: int, 
#                          value_prob: float, threshold: int = 1, max_iter: int = 1000): 
#     """
#     A sub-population is selected at random taking in consideration the arguments.
    
#     @param df_train: dataframe containing the train population
#     @param df_test: dataframe containing the test population
#     @param domains: mapping between covariates and their possible values
#     @param num_cov: number of covariates to be included in the affected sub-population
#     @param value_prob: probability of affecting a value of a covariate
#     @param threshold: minimum number of rows in a sub-population
#     @param max_iter: maximum number of iteration to search a population. If exceeded, return None
#     :return: a new dataframe that contains the bias sub-population, sub-population affected
#     """
    
#     # generate affected sup-population
#     for i in count():
#         cov_affected = np.random.choice(list(domains.keys()), size=num_cov, replace=False)

#         # for each covariate affected, choose with prob value_prob the values that are affected
#         s_affected = dict()

#         for cov in cov_affected:
#             mask = np.random.rand(len(domains[cov])) < value_prob

#             # if every covariate value was picked, remove one
#             # if none of the covariate value was picked, pick one
#             if np.all(mask) or not np.any(mask):
#                 index = np.random.randint(0, len(mask))
#                 mask[index] = 0 if np.all(mask) else 1

#             s_affected[cov] = domains[cov][mask]

#         # create mask of affected sub-population
#         mask_train = np.ones((df_train.shape[0], )).astype(np.bool)
#         mask_test = np.ones((df_test.shape[0], )).astype(np.bool)
        
#         for cov in s_affected:
#             mask_train &= np.isin(df_train[cov].to_numpy(), s_affected[cov])
#             mask_test  &= np.isin(df_test[cov].to_numpy(), s_affected[cov])
        
#         # check if there is any entry corresponding to the affected sub-population
#         # if not, repeat the generating process
#         if np.sum(mask_train) > threshold and np.sum(mask_test) > threshold:
#             break
        
#         if i > max_iter:
#             return None
        
#     return s_affected


# # In[ ]:


# def bias_subpopulation(df: pd.DataFrame, s_affected: dict, ys_cname: str, q: float):
#     """
#     The sub-population passed as argument is biased
    
#     @param df: dataframe containing the entier initially unbiased population
#     @param s_affected: sub-population to be biased
#     @param ys_cname: outcome column name
#     @param q: risk factor
#     :return: the entier population containing the biased sub-population
#     """
#     sub_df = MDSS.get_subpopulation(df, s_affected)
#     indexes = sub_df.index
    
#     # generate outcome corresponding to H1
#     mean_y = df[ys_cname].to_numpy()[indexes].mean()
#     p = q * mean_y / (1 - mean_y + q * mean_y)
#     df.loc[indexes, ys_cname] = (np.random.rand(len(indexes)) < p).astype(np.int)
#     return df


# # ### Test 3

# # Covariate shift.  This is where we also reduce the number of rows -- but not uniformly!  We pick a random subset (female, black or other) and then we reduce the number of rows in the training set that correspond to that subset.  This is different from 1 above because it changes the joint distribution of X's because we are removing only certain types of records. At first, only 1% of that subset is kept. Then 2% of that subset. Then 4%, 8%, 16%, 32% of that subset.

# # In[ ]:


# def experiment3(s_affected: dict, p: int, i: int, seed: int):
#     """
#     @param s_affected: sub-population from which we sample
#     @param p: percentage of that sub-population to be sampled
#     @param i: experiment index
#     @param seed: experiment seed
#     """
#     np.random.randint(seed)
    
#     # extract rows containing that sub-population
#     sub_mdscan_data_train = MDSS.get_subpopulation(mdscan_data_train, s_affected)
    
#     # extract the indexes corresponding to that sub-population (sub_indexes)
#     # and all the other indexes that correspond to rows that are not part
#     # of that sub-population (compl_sub_indexes)
#     sub_indexes = set(sub_mdscan_data_train.index)
#     compl_sub_indexes = set(mdscan_data_train.index).difference(sub_indexes)

#     # select at random with replacement p% from the sub_indexes
#     num_rows = len(sub_indexes)
#     sub_indexes = np.random.choice(list(sub_indexes), size=num_rows * p // 100, replace=True)

#     # get final indexes as the concatentaion of the complement indexes and the p percentage of indexes
#     indexes = list(compl_sub_indexes) + list(sub_indexes)
#     sub_train_data = train_data.loc[indexes, :].to_numpy()

#     # first column is the outcome
#     X_train, y_train = sub_train_data[:, 1:], sub_train_data[:, 0]
#     X = test_data.to_numpy()[:, 1:]
    
#     for clf_label in classifiers_labels:
#         # train the classifier on smaller dataset
#         clf = model_selection(X=X_train, y=y_train, clf_label=clf_label)
#         clf = clf.fit(X_train, y_train)

#         # predict the probability on the entier datast
#         proba = clf.predict_proba(X)[:, 1]
       
#         # save probability in a file
#         filename = "training/test3/clf=%s_p=%d_i=%d.pkl" % (clf_label, p, i)
#         file = open(filename, 'wb')
#         pkl.dump(proba, file)
#         file.close()


# # In[ ]:


# percentages = [1, 2, 4, 8, 16, 32]
# num_experiments = NUM_EXPERIMENTS

# # transform categorical variables into dummy
# train_data = pd.get_dummies(mdscan_data_train, prefix=covariates, columns=covariates)
# test_data = pd.get_dummies(mdscan_data_test, prefix=covariates, columns=covariates)

# # list of affected sub-populations
# ss_affected = []

# for i in range(num_experiments):
#     # select random sub-population
#     s_affected = random_subpopulation(df_train=mdscan_data_train, df_test=mdscan_data_test, 
#                                       domains=domains, num_cov=3, value_prob=0.5, threshold=100)
#     assert s_affected is not None, "Couldn't find any sub-population"
#     ss_affected.append(s_affected)
    
# # start time
# start = time.time()
    
# # define thread pool
# pool = Pool(processes=NUM_CPUS)

# for p in percentages:
#     seeds = np.random.randint(0, 99999, size=num_experiments)    
#     for i, s_affected in enumerate(ss_affected):
#         pool.apply_async(experiment3, (s_affected, p, i, seeds[i]))
        
# # close thread pool & wait for all jobs to be done
# pool.close()
# pool.join()

# # print duration
# print("Ellapsed: %.2f" % (time.time() - start))


# # ### Test 4

# # Transfer learning.  Here we also pick a random subset apriori.  But instead of reducing the number of rows in the training data that correspond to that group (which 3 does), we now change the Y outcomes from this particular subgroup BEFORE training.  This means the model is learning a different Y|X (outcomes conditioned on X) for that subgroup.  So when we compare it back to the original unaltered data for scanning, we should see the bias induced by these changes in Y. 

# # In[ ]:


# def experiment4(s_affected, q, i, seed):
#     np.random.seed(seed)

#     score_results = {}
#     accuracy_results = {}
    
#     # introduce bias in that sub-population
#     mdscan_data_train_ = mdscan_data_train.copy()
#     mdscan_data_train_ = bias_subpopulation(df=mdscan_data_train_, s_affected=s_affected, 
#                                           ys_cname="outcome", q=q)
#     train_data = pd.get_dummies(mdscan_data_train_, prefix=covariates, columns=covariates)
    
#     # first column is the outcome
#     X_train, y_train = train_data.to_numpy()[:, 1:], train_data.to_numpy()[:, 0]
#     X = test_data.to_numpy()[:, 1:]

#     for clf_label in classifiers_labels:
#         # train the classifier on smaller dataset
#         clf = model_selection(X=X_train, y=y_train, clf_label=clf_label)
#         clf = clf.fit(X_train, y_train)

#         # predict the probability on the entier datast
#         proba = clf.predict_proba(X)[:, 1]
        
#         # save probability into a file
#         filename = "./training/test4/clf=%s_q=%.2f_i=%d.pkl" % (clf_label, q, i)
#         file = open(filename, 'wb')
#         pkl.dump((proba, s_affected), file)
#         file.close()


# # In[ ]:


# qs = [.9, .7, .5, .3, .1]
# num_experiments = NUM_EXPERIMENTS

# # transform categorical variables into dummy

# test_data = pd.get_dummies(mdscan_data_test, prefix=covariates, columns=covariates)

# # list of affected sub-populations
# ss_affected = []

# for i in range(num_experiments):
#     # select random sub-population
#     s_affected = random_subpopulation(df_train=mdscan_data_train, df_test=mdscan_data_test, domains=domains, 
#                                       num_cov=3, value_prob=0.5, threshold=100)
#     assert s_affected is not None, "Couldn't find any sub-population"
#     ss_affected.append(s_affected)
    
# # start time
# start = time.time()
    
# # define thread pool
# pool = Pool(processes=NUM_CPUS)

# for q in qs:
#     seeds = np.random.randint(0, 99999, size=num_experiments)    
#     for i, s_affected in enumerate(ss_affected):
#         pool.apply_async(experiment4, (s_affected, q, i, seeds[i]))
        
# # close thread pool & wait for all jobs to be done
# pool.close()
# pool.join()

# # print duration
# print("Ellapsed: %.2f" % (time.time() - start))


# ### Test 5

# Eliminate columns one at the time to see the bias effect induced by removing a feature.

# In[ ]:


def experiment5(cov, i, seed):
    np.random.seed(seed)
    
    results = {}
    num_rows = mdscan_data_train.shape[0]
    mdscan_data_train_ = mdscan_data_train.copy()
    mdscan_data_test_ = mdscan_data_test.copy()
    
    # drop covariates from dataframe
    sub_mdscan_data_train = mdscan_data_train_.drop([cov], axis=1) if cov else mdscan_data_train_
    sub_mdscan_data_test = mdscan_data_test_.drop([cov], axis=1) if cov else mdscan_data_test_
    
    # convert sub_mdscan_data to one hot encodings / dummy variables
    columns = list(set(covariates).difference(set([cov]))) if cov else covariates
    
    train_data = pd.get_dummies(sub_mdscan_data_train, prefix=columns, columns=columns)
    indexes = np.random.choice(np.arange(num_rows), num_rows // 2, replace=False)
    sub_train_data = train_data.loc[indexes, :].to_numpy()
    test_data = pd.get_dummies(sub_mdscan_data_test, prefix=columns, columns=columns)
    
    # first column is the outcome
    X_train, y_train = sub_train_data[:, 1:], sub_train_data[:, 0]
    X = test_data.to_numpy()[:, 1:]

    for clf_label in classifiers_labels:
        # train the classifier on smaller dataset
        clf = model_selection(X=X_train, y=y_train, clf_label=clf_label)
        clf = clf.fit(X_train, y_train)

        # predict the probability on the entier datast
        proba = clf.predict_proba(X)[:, 1]
           
        # save probability into a file
        filename = "training/test5/clf=%s_cov=%s_i=%d.pkl" % (clf_label, cov, i)
        file = open(filename, 'wb')
        pkl.dump(proba, file)
        file.close()


# In[ ]:


num_experiments = NUM_EXPERIMENTS

# start time
start = time.time()

# define thread pool
pool = Pool(processes=NUM_CPUS)
extended_cov = [None] #+ covariates
for cov in extended_cov:   
    seeds = np.random.randint(0, 99999, size=num_experiments)
    for i in range(num_experiments):
        pool.apply_async(experiment5, (cov, i, seeds[i]))
        
# close thread pool & wait for all jobs to be done
pool.close()
pool.join()

# print duration
print("Ellapsed: %.2f" % (time.time() - start))

