# -*- coding: utf-8 -*-
# ----------------------------------------------#
# OverRule: Overlap Estimation using Rule Sets  #
# @Authors: Dennis Wei                          #
# ----------------------------------------------#

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

    
from typing import Optional, Tuple, Union
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

###########################################
#%% Top-level function
def load_process_data(filePath, rowHeader, colNames, colSep=',', fracPresent=0.9, col_y=None, valEq_y=None, colCateg=[], numThresh=9, negations=False):
    ''' Load CSV file and process data for BCS rule-learner

    Inputs:
    filePath = full path to CSV file
    rowHeader = index of row containing column names or None
    colNames = column names (if none in file or to override)
    colSep = column separator
    fracPresent = fraction of non-missing values needed to use a column (default 0.9)
    col_y = name of target column
    valEq_y = value to test for equality to binarize non-binary target column
    colCateg = list of names of categorical columns
    numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
    negations = whether to append negations

    Outputs:
    A = binary feature DataFrame
    y = target column'''

    # Read CSV file
    data = pd.read_csv(filePath, sep=colSep, names=colNames, header=rowHeader, error_bad_lines=False)

    # Remove columns with too many missing values
    data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
    # Remove rows with any missing values
    data.dropna(axis=0, how='any', inplace=True)

    # Extract and binarize target column
    y = extract_target(data, col_y, valEq_y)

    # Binarize features
    A = binarize_features(data, colCateg, numThresh, negations)

    return A, y

#%% Extract and binarize target variable
def extract_target(data, col_y=None, valEq_y=None, valGt_y=None, **kwargs):
    '''Extract and binarize target variable

    Inputs:
    data = original feature DataFrame
    col_y = name of target column
    valEq_y = values corresponding to y = 1 for binarizing target
    valGt_y = threshold for binarizing target, above which y = 1

    Output:
    y = target column'''

    ### dmm: if no col_y specified -- use the last
    if not col_y and (col_y != 0):
        col_y = data.columns[-1]
    # Separate target column
    y = data.pop(col_y)
    if valEq_y or valEq_y == 0:
        # Binarize if values for equality test provided
        if type(valEq_y) is not list:
            valEq_y = [valEq_y]
        y = y.isin(valEq_y).astype(int)
    elif valGt_y or valGt_y == 0:
        # Binarize if threshold for comparison provided
        y = (y > valGt_y).astype(int)
    # Ensure y is binary and contains no missing values
    assert y.nunique() == 2, "Target 'y' must be binary"
    assert y.count() == len(y), "Target 'y' must not contain missing values"
    # Rename values to 0, 1
    y.replace(np.sort(y.unique()), [0, 1], inplace=True)

    return y

#%% Binarize features
def binarize_features(data, colCateg=[], numThresh=9, negations=False, threshStr=False, **kwargs):
    '''Binarize categorical and ordinal (including continuous) features

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)
    numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
    negations = whether to append negations
    threshStr = whether to convert thresholds on ordinal features to strings

    Outputs:
    A = binary feature DataFrame'''

    # Quantile probabilities
    quantProb = np.linspace(1./(numThresh + 1.), numThresh/(numThresh + 1.), numThresh)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize dataframe and thresholds
    A = pd.DataFrame(index=data.index,
                     columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}

    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[(str(c), '', '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
            if negations:
                A[(str(c), 'not', '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            # Dummy-code values
            if data[c].dtype == float:
                Anew = pd.get_dummies(data[c].astype(str)).astype(int)
            else:
                Anew = pd.get_dummies(data[c]).astype(int)
            Anew.columns = Anew.columns.astype(str)
            if negations:
                # Append negations
                Anew = pd.concat([Anew, 1-Anew], axis=1, keys=[(str(c),'=='), (str(c),'!=')])
            else:
                Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        # Ordinal column
        elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
            | np.issubdtype(data[c].dtype, np.dtype(float).type):
            # Few unique values
            if valUniq <= numThresh + 1:
                # Thresholds are sorted unique values excluding maximum
                thresh[c] = np.sort(data[c].unique())[:-1]
            # Many unique values
            else:
                # Thresholds are quantiles excluding repetitions
                thresh[c] = data[c].quantile(q=quantProb).unique()
            # Threshold values to produce binary arrays
            Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            if negations:
                # Append negations
                Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                ops = ['<=', '>']
            else:
                ops = ['<=']
            # Convert to dataframe with column labels
            if threshStr:
                Anew = pd.DataFrame(Anew, index=data.index,
                                    columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]))
            else:
                Anew = pd.DataFrame(Anew, index=data.index,
                                    columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]]))
            indNull = data[c].isnull()
            if indNull.any():
                # Ensure that rows corresponding to NaN values are zeroed out
                Anew.loc[indNull] = 0
                # Add NaN indicator column
                Anew[(str(c), '==', 'NaN')] = indNull.astype(int)
                if negations:
                    Anew[(str(c), '!=', 'NaN')] = (~indNull).astype(int)
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        else:
            print(("Skipping column '" + str(c) + "': data type cannot be handled"))
            continue

    return A

def binarize_categ(data, colCateg=[], **kwargs):
    '''Binarize categorical features only

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)

    Outputs:
    A = numeric feature DataFrame'''

    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize dataframe and thresholds
    A = pd.DataFrame(index=data.index)

    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Constant column --- discard
        if valUniq < 2:
            continue

        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A[str(c)] = data[c].replace(np.sort(data[c].unique()), [0, 1])

        # Categorical column
        elif (c in colCateg) or (data[c].dtype == 'object'):
            # Dummy-code values
            if data[c].dtype == float:
                Anew = pd.get_dummies(data[c].astype(str)).astype(int)
            else:
                Anew = pd.get_dummies(data[c]).astype(int)
            Anew.columns = str(c) + '==' + Anew.columns.astype(str)
            # Concatenate
            A = pd.concat([A, Anew], axis=1)

        # Ordinal column
        elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
            | np.issubdtype(data[c].dtype, np.dtype(float).type):
            # Leave as is
            A[str(c)] = data[c]

        else:
            print(("Skipping column '" + str(c) + "': data type cannot be handled"))
            continue

    return A

class FeatureBinarizer(TransformerMixin):
    '''Transformer for binarizing categorical and ordinal (including continuous) features
        Parameters:
            colCateg = list of categorical features ('object' dtype automatically treated as categorical)
            numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
            negations = whether to append negations
            threshStr = whether to convert thresholds on ordinal features to strings
            threshOverride = dictionary of {colname : np.linspace object} to define cuts
    '''
    def __init__(self, colCateg=[], numThresh=9, negations=False,
            threshStr=False, threshOverride={}, **kwargs):
        # List of categorical columns
        if type(colCateg) is pd.Series:
            self.colCateg = colCateg.tolist()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg

        self.threshOverride = {} if threshOverride is None else threshOverride
        # Number of quantile thresholds used to binarize ordinal features
        self.numThresh = numThresh
        self.thresh = {}
        # whether to append negations
        self.negations = negations
        # whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr

    def fit(self, X):
        '''Inputs:
            X = original feature DataFrame
        Outputs:
            maps = dictionary of mappings for unary/binary columns
            enc = dictionary of OneHotEncoders for categorical columns
            thresh = dictionary of lists of thresholds for ordinal columns
            NaN = list of ordinal columns containing NaN values'''
        data = X
        # Quantile probabilities
        quantProb = np.linspace(1. / (self.numThresh + 1.), self.numThresh / (self.numThresh + 1.), self.numThresh)
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        NaN = []

        # Iterate over columns
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(data[c].unique()))

            # Categorical column
            elif (c in self.colCateg) or (data[c].dtype == 'object'):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(data[[c]])

            # Ordinal column
            elif np.issubdtype(data[c].dtype, np.dtype(int).type) \
                | np.issubdtype(data[c].dtype, np.dtype(float).type):
                # Few unique values
                if valUniq <= self.numThresh + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                elif c in self.threshOverride.keys():
                    thresh[c] = self.threshOverride[c]
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
                if data[c].isnull().any():
                    # Contains NaN values
                    NaN.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.NaN = NaN
        return self

    def transform(self, X):
        '''Inputs:
            X = original feature DataFrame
        Outputs:
            A = binary feature DataFrame'''
        data = X
        maps = self.maps
        enc = self.enc
        thresh = self.thresh
        NaN = self.NaN

        # Initialize dataframe
        A = pd.DataFrame(index=data.index,
                         columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))

        # Iterate over columns
        for c in data:
            # Constant or binary column
            if c in maps:
                # Rename values to 0, 1
                A[(str(c), '', '')] = data[c].map(maps[c])
                if self.negations:
                    A[(str(c), 'not', '')] = 1 - A[(str(c), '', '')]

            # Categorical column
            elif c in enc:
                # Apply OneHotEncoder
                Anew = enc[c].transform(data[[c]])
                Anew = pd.DataFrame(Anew, index=data.index, columns=enc[c].categories_[0].astype(str))
                if self.negations:
                    # Append negations
                    Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(str(c), '=='), (str(c), '!=')])
                else:
                    Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif c in thresh:
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                if self.negations:
                    # Append negations
                    Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                    ops = ['<=', '>']
                else:
                    ops = ['<=']
                # Convert to dataframe with column labels
                if self.threshStr:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c].astype(str)]))
                else:
                    Anew = pd.DataFrame(Anew, index=data.index,
                                        columns=pd.MultiIndex.from_product([[str(c)], ops, thresh[c]]))
                if c in NaN:
                    # Ensure that rows corresponding to NaN values are zeroed out
                    indNull = data[c].isnull()
                    Anew.loc[indNull] = 0
                    # Add NaN indicator column
                    Anew[(str(c), '==', 'NaN')] = indNull.astype(int)
                    if self.negations:
                        Anew[(str(c), '!=', 'NaN')] = (~indNull).astype(int)
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        return A

#%% Discretize continuous features and standardize values
def bin_cont_features(data, colCateg=[], numThresh=9, **kwargs):
    '''Bin continuous features using quantiles

    Inputs:
    data = original feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)
    numThresh = number of quantile thresholds not including min/max (default 9)

    Outputs:
    A = discretized feature DataFrame'''

    # Quantile probabilities
    quantProb = np.linspace(0., 1., numThresh+2)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    # Initialize DataFrame
    A = data.copy()
    # Iterate over columns
    for c in data:
        # number of unique values
        valUniq = data[c].nunique()

        # Only bin non-categorical numerical features with enough unique values
        if (np.issubdtype(data[c].dtype, np.dtype(int).type) \
        or np.issubdtype(data[c].dtype, np.dtype(float).type))\
        and (c not in colCateg) and valUniq > numThresh + 1:
            A[c] = pd.qcut(A[c], q=quantProb, duplicates='drop')
            if A[c].nunique() == 1:
                # Collapsed into single bin, re-separate into two bins
                quant = data[c].quantile([0, 0.5, 1])
                quant[0] -= 1e-3
                if quant[0.5] == quant[1]:
                    quant[0.5] -= 1e-3
                A[c] = pd.cut(data[c], quant)

    return A

def std_values(data, colCateg=[], **kwargs):
    '''Standardize values of (already discretized) features

    Inputs:
    data = input feature DataFrame
    colCateg = list of categorical features ('object' dtype automatically treated as categorical)

    Outputs:
    A = standardized feature DataFrame
    mappings = dictionary of value mappings'''

    # Initialize
    A = data.copy()
    mappings = {}
    isCategory = A.dtypes == 'category'
    # Iterate over columns
    for c in A:
        # number of unique values
        valUniq = A[c].nunique()

        # Binned numerical column, which has 'category' dtype
        if isCategory[c]:
            # Map bins to integers
            mappings[c] = pd.Series(range(valUniq), index=A[c].cat.categories)
            A[c].cat.categories = range(valUniq)

        # Binary column
        elif valUniq == 2:
            # Map sorted values to 0, 1
            mappings[c] = pd.Series([0, 1], index=np.sort(A[c].dropna().unique()))
            A[c] = A[c].map(mappings[c])

        # Categorical column
        elif (c in colCateg) or (A[c].dtype == 'object'):
            # First map sorted values to integers
            mappings[c] = pd.Series(range(valUniq), index=np.sort(A[c].dropna().unique()))
            # Then map to alphabetic encoding of integers
            mappings[c] = mappings[c].map(digit_to_alpha)
            A[c] = A[c].map(mappings[c])

        # Non-binned numerical column (because it has few unique values)
        elif np.issubdtype(A[c].dtype, np.dtype(int).type) \
            or np.issubdtype(A[c].dtype, np.dtype(float).type):
            # Map sorted values to integers
            mappings[c] = pd.Series(range(valUniq), index=np.sort(A[c].dropna().unique()))
            A[c] = A[c].map(mappings[c])

    return A, mappings

def digit_to_alpha(n):
    '''Map digits in integer n to letters
    0 -> A, 1 -> B, 2 -> C, ..., 9 -> J'''
    return ''.join([chr(int(d) + ord('A')) for d in str(n)])

#%% Split data into training and test sets
def split_train_test(A, y, dirData, fileName, numFold=10, numRepeat=10, concatMultiIndex=False):
    '''Split data into training and test sets using repeated stratified K-fold CV
    and save as CSV

    Inputs:
    A = binary feature DataFrame
    y = target column
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    numFold = number of folds (K)
    numRepeat = number of K-fold splits
    concatMultiIndex = whether to concatenate column MultiIndex into single level

    Output: total number of splits = numFold x numRepeat'''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(y.name,'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)
    # Iterate over splits
    rskf = RepeatedStratifiedKFold(n_splits=numFold, n_repeats=numRepeat)
    for (split, (idxTrain, idxTest)) in enumerate(rskf.split(A, y)):
        # Save training and test sets as CSV
        filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
        B.iloc[idxTrain].to_csv(filePath + 'train.csv', index=False)
        B.iloc[idxTest].to_csv(filePath + 'test.csv', index=False)

    return rskf.get_n_splits()

def save_train_test(A, y, splits, dirData, fileName, concatMultiIndex=False):
    '''Save training and test sets as CSV if given splits, otherwise full dataset

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    concatMultiIndex = whether to concatenate column MultiIndex into single level
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        if concatMultiIndex == 'BOA':
            # Special formatting for Wang et al.'s Bayesian Rule Sets
            B.columns = B.columns.get_level_values(0).str.replace('_','-') + '_'\
            + B.columns.get_level_values(1) + B.columns.get_level_values(2)
        else:
            B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)

    if splits:
        for (split, (idxTrain, idxTest)) in enumerate(splits):
            # Save training and test sets as CSV
            filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
            B.iloc[idxTrain].to_csv(filePath + 'train.csv', index=False)
            B.iloc[idxTest].to_csv(filePath + 'test.csv', index=False)
    else:
        # Save full dataset
        filePath = os.path.join(dirData, fileName + '.csv')
        B.to_csv(filePath, index=False)

    return

def pickle_train_test(A, y, splits, dirData, fileName):
    '''Pickle training and test sets if given splits, otherwise full dataset

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y

    if splits:
        for (split, (idxTrain, idxTest)) in enumerate(splits):
            # Pickle training and test sets
            filePath = os.path.join(dirData, fileName + '_' + format(split, '03d') + '_')
            B.iloc[idxTrain].to_pickle(filePath + 'train.pkl')
            B.iloc[idxTest].to_pickle(filePath + 'test.pkl')
    else:
        # Pickle full dataset
        filePath = os.path.join(dirData, fileName + '.pkl')
        B.to_pickle(filePath)

    return

def save_internal_train(A, y, splits, dirData, fileName, concatMultiIndex=False):
    '''Save internal training sets as CSV for parameter selection

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    concatMultiIndex = whether to concatenate column MultiIndex into single level
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y
    # Concatenate column MultiIndex into single level
    if concatMultiIndex and (type(B.columns) is pd.MultiIndex):
        if concatMultiIndex == 'BOA':
            # Special formatting for Wang et al.'s Bayesian Rule Sets
            B.columns = B.columns.get_level_values(0).str.replace('_','-') + '_'\
            + B.columns.get_level_values(1) + B.columns.get_level_values(2)
        else:
            B.columns = B.columns.get_level_values(0) + B.columns.get_level_values(1) + B.columns.get_level_values(2)

    # Iterate over test set index
    numSplit = len(splits)
    with open(os.path.join(dirData, 'splitsInt.txt'), 'w') as f:
        for (test, (idxTrain, idxTest)) in enumerate(splits):
            # Iterate over validation set index
            for valid in range(test+1, numSplit):
                # Internal training set indices
                idxIntTrain = np.setdiff1d(idxTrain, splits[valid][1])
                # Save as CSV
                filePath = os.path.join(dirData, fileName + '_' + format(test, '03d') + '_'\
                                        + format(valid, '03d') + '_train.csv')
                B.iloc[idxIntTrain].to_csv(filePath, index=False)
                # Write training, test, and validation indices to text file
                f.write(str(idxIntTrain).strip('[]').replace('\n','') + '\n')
                f.write(str(idxTest).strip('[]').replace('\n','') + '\n')
                f.write(str(splits[valid][1]).strip('[]').replace('\n','') + '\n')

    return

def pickle_internal_train(A, y, splits, dirData, fileName):
    '''Pickle internal training sets for parameter selection

    Inputs:
    A = feature DataFrame
    y = target column
    splits = list of training and test set indices
    dirData = directory where training and test sets will be saved
    fileName = dataset name to be used as root of filenames
    '''

    # Append target as last column
    B = A.copy()
    if type(B.columns) is pd.MultiIndex:
        B[(str(y.name),'','')] = y
    else:
        B[y.name] = y

    # Iterate over test set index
    numSplit = len(splits)
    for (test, (idxTrain, idxTest)) in enumerate(splits):
        # Iterate over validation set index
        for valid in range(test+1, numSplit):
            # Internal training set indices
            idxIntTrain = np.setdiff1d(idxTrain, splits[valid][1])
            # Save as CSV
            filePath = os.path.join(dirData, fileName + '_' + format(test, '03d') + '_'\
                                    + format(valid, '03d') + '_train.pkl')
            B.iloc[idxIntTrain].to_pickle(filePath)

    return


#%% Call function if run as script
##########################################
def example():

    # File parameters
    workDir = u''
    fileDir = './Data/' #u'\\Data\\'
    fileName = u'iris_categ.csv'
    colSep = ','
    rowHeader = None
    colNames = ['X1','X2','sepal length','sepal width','petal length','petal width','X7','X8','iris species']
    col_y = colNames[-1]
    colCateg = 'X8' # ['X8']

    A, y = load_process_data(workDir + fileDir + fileName, rowHeader, colNames, colSep=colSep, col_y=col_y, valEq_y=2, colCateg=colCateg)
###########################################
def example2():

    fname_data = 'Data/iris_bin.csv'
    colSep = ','
    rowHeader = None
    colNames = ['X1', 'X2', 'sepal length', 'sepal width', 'petal length', 'petal width', 'iris species']
    col_y = colNames[-1]
    A, y = load_process_data(fname_data, rowHeader, colNames, colSep=colSep, col_y=col_y, valEq_y=2)

###########################################
if __name__ == '__main__':
    #example2()
    example()


# noinspection PyPep8Naming
class FeatureBinarizerFromTrees(TransformerMixin):
    """Transformer for binarizing categorical and ordinal features.
    For use with BooleanRuleCG, LogisticRuleRegression, and LinearRuleRegression. This transformer generates binary
    features using splits in decision trees. Compared to `FeatureBinarizer`, this approach reduces the number of
    features required to produce an accurate model. The smaller feature space shortens training time and often
    simplifies rule sets.
    """

    # Listing members here to provide type hints and facilitate code inspection and self-documentation.
    # The names follow FeatureBinarizer
#     colCateg: list
#     enc: dict
#     features: pd.MultiIndex
#     maps: dict
#     ordinal: list
#     randomState: int
#     returnOrd: bool
#     scaler: StandardScaler
#     thresh: dict
#     threshRound: Union[int, None]
#     threshStr: bool
#     treeDepth: Union[int, None]
#     treeFeatureSelection: Union[str, float, None]
#     treeKwargs: dict
#     treeNum: int

    def __init__(self,
                 colCateg: list = None,
                 treeNum: int = 2,
                 treeDepth: Optional[int] = 4,
                 treeFeatureSelection: Union[str, float, None] = None,
                 treeKwargs: dict = None,
                 threshRound: Optional[int] = 6,
                 threshStr: bool = False,
                 returnOrd: bool = False,
                 randomState: int = None,
                 **kwargs):
        """
        Args:
            colCateg (list): Categorical features ('object' dtype automatically treated as categorical). These features
                are one-hot-encoded.
            treeNum (int): Number of trees to fit. Setting 'treeNum' to a value greater than one usually produces a
                larger variety of output features.
            treeDepth (int): The maximum depth of the tree. Setting 'treeDepth=None' grows a tree without limit.
                Larger depth values produce more output features. Corresponds to parameter 'max_depth' in
                DecisionTreeClassifier.
            treeFeatureSelection (float, str): When building a tree, the input features are randomly permuted at
                each split. This parameter specifies how many input features are considered at each split. By default,
                this parameter is set to 'None' which indicates that all features should be considered at every split.
                Other possible values are 'sqrt', 'log2', or a float that indicates the proportion of features to
                select at every split (e.g. 0.5 would randomly select half of the input features at every split).
                To create a wide variety of output features, or to sift through a very large number of features,
                increase 'treeNum' and set 'treeFeatureSelection="sqrt"'. Corresponds to 'max_features' in
                DecisionTreeClassifier.
            treeKwargs (dict): A dictionary of parameters to pass to the scikit-learn DecisionTreeClassifier during
                fitting.
            threshRound (int): Round threshold values by this number of decimal places. This parameter can be used
                to prevent similar thresholds from generating separate binarized features. E.g., if 'threshRound=2',
                only one binarized feature will be generated for thresholds 0.009 and 0.01. Setting 'threshRound=None'
                will disable rounding.
            threshStr (bool): Convert threshold values to strings, including categorical values, in transformed
                data frame's index.
            returnOrd (bool): Return a standardized data frame for ordinal features (both discrete and continuous)
                during transformation in addition to the binarized data frame.
            randomState (int): Random state for decision tree.
        """

        # Categorical columns
        if colCateg is None:
            self.colCateg = []
        elif type(colCateg) is Series:
            self.colCateg = colCateg.to_list()
        elif type(colCateg) is not list:
            self.colCateg = [colCateg]
        else:
            self.colCateg = colCateg

        # Number of trees
        if (treeNum is None) or (treeNum < 1) or (int(treeNum) != treeNum):
            raise ValueError('The value for \'treeNum\' must be an integer value greater than zero.')
        self.treeNum = int(treeNum)

        # Tree kwargs
        if treeKwargs is None:
            treeKwargs = dict(max_features=None)
        elif 'max_features' not in treeKwargs:
            treeKwargs['max_features'] = None

        # Tree depth
        if treeDepth is not None:
            if (treeDepth < 1) or (int(treeDepth) != treeDepth):
                raise ValueError('The value for \'treeDepth\' must be None or an integer value greater than zero.')
            treeKwargs['max_depth'] = treeDepth
        elif 'max_depth' in treeKwargs:
            treeDepth = treeKwargs['max_depth']
        self.treeDepth = treeDepth

        # Tree feature selection
        if treeFeatureSelection is not None:
            if isinstance(treeFeatureSelection, str):
                error = treeFeatureSelection not in ('auto', 'sqrt', 'log2')
            elif isinstance(treeFeatureSelection, (float, int)):
                error = (treeFeatureSelection <= 0.0) or (treeFeatureSelection > 1.0)
            else:
                error = True
            if error:
                raise ValueError('Valid values for \'treeFeatureSelection\' are None, \'auto\', \'sqrt\', \'log2\', or '
                                 'a value in interval (0, 1].')
            treeKwargs['max_features'] = treeFeatureSelection
        elif 'max_features' in treeKwargs:
            treeFeatureSelection = treeKwargs['max_features']
        self.treeFeatureSelection = treeFeatureSelection

        # Tree kwargs
        self.treeKwargs = treeKwargs

        # Random state
        self.randomState = randomState

        # Rounding for ordinal values
        if (threshRound is not None) and (threshRound < 0):
            raise ValueError('The value for \'threshRound\' must be None, or zero, or greater than zero.')
        self.threshRound = threshRound

        # Whether to convert thresholds on ordinal features to strings
        self.threshStr = threshStr

        # Whether to convert thresholds on ordinal features to strings.  Also return standardized ordinal features
        # during transformation
        self.returnOrd = returnOrd

    def _fit_transform_like_feature_binarizer(self, X: DataFrame) -> DataFrame:
        # Initialize
        maps = {}
        enc = {}
        thresh = {}
        ordinal = []
        A = DataFrame(index=X.index,
                      columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))

        # Iterate over columns
        for c in X:
            # number of unique values
            valUniq = X[c].nunique()

            # Constant or binary column
            if valUniq <= 2:
                # Mapping to 0, 1
                maps[c] = pd.Series(range(valUniq), index=np.sort(X[c].unique()))
                A[(str(c), '', '')] = X[c].map(maps[c])

            # Categorical column
            elif (c in self.colCateg) or (X[c].dtype == 'object'):
                # OneHotEncoder object
                enc[c] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
                # Fit to observed categories
                enc[c].fit(X[[c]])
                # Apply OneHotEncoder
                Anew = enc[c].transform(X[[c]])
                # Original FeatureBinarizer converts all values to str. This class preserves type to be used
                # during transform.
                Anew = DataFrame(Anew, index=X.index, columns=enc[c].categories_[0])
                Anew.columns = pd.MultiIndex.from_product([[str(c)], ['=='], Anew.columns])
                # Concatenate
                A = pd.concat([A, Anew], axis=1)

            # Ordinal column
            elif np.issubdtype(X[c].dtype, np.integer) | np.issubdtype(X[c].dtype, np.floating):
                # Unlike FeaturBinarizer, just append the original ordinal column. It will be fit by the
                # DecisionTreeClassifier.
                Anew = DataFrame(
                    X[c].to_numpy(),  # Required
                    columns=pd.MultiIndex.from_arrays([[c], ['<='], [0.0]], names=['feature', 'operation', 'value']),
                    index=X.index
                )
                A = pd.concat([A, Anew], axis=1)
                ordinal.append(c)

            else:
                print(("Skipping column '" + str(c) + "': data type cannot be handled"))
                continue

        self.maps = maps
        self.enc = enc
        self.thresh = thresh
        self.ordinal = ordinal

        return A

    def fit(self, X: DataFrame, y: Union[ndarray, DataFrame, Series, list] = None):
        """Fit transformer. NaN/None values are not permitted for X or y.
        Args:
            X (DataFrame): Original features
            y (Iterable): Target
        Returns:
            FeatureBinarizerFromTrees: Self
            self.enc (dict): OneHotEncoders for categorical columns
            self.features (MultiIndex): Pandas MultiIndex of feature names, operations, and values
            self.maps (dict): Mappings for unary/binary columns
            self.ordinal (list): Ordinal columns
            self.scaler (StandardScaler): StandardScaler for ordinal columns
            self.thresh (dict(array)): Thresholds for ordinal columns
        """

        # The decision tree will also throw an exception, but it is cryptic.
        if y is None:
            raise ValueError('The parameter \'y\' is required.')

        # Binarize unary/binary/categorical features according to the FeatureBinarizer style. Ordinal columns
        # are not binarized: i.e., they are included as-is in the data frame with '<=' operations in the index.
        # They will be binarized by extracting splits from the decision tree.
        Xfb = self._fit_transform_like_feature_binarizer(X)

        # Fit decision trees.
        featuresIdx = np.empty((0,), int)
        thresholds = np.empty((0,), int)

        randomState = self.randomState
        for i in range(self.treeNum):
            if randomState is not None:
                randomState += i
            tree = DecisionTreeClassifier(random_state=randomState, **self.treeKwargs)
            tree.fit(Xfb, y)
            featuresIdx = np.hstack((featuresIdx, tree.tree_.feature))
            thresholds = np.hstack((thresholds, tree.tree_.threshold))

        # See `Understanding the decision tree structure`
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        select = featuresIdx != -2  # Indicates leaf nodes
        featuresIdx = featuresIdx[select]
        thresholds = thresholds[select]

        # Important to do rounding before dropping duplicates because rounding can create duplicates.
        if (self.threshRound is not None) and len(self.ordinal):
            thresholds = thresholds.round(self.threshRound)

        # Create frame from column index which contains the relevant features
        features = Xfb.columns[featuresIdx].to_frame(False)

        # Set thresholds for ordinal values
        if len(self.ordinal):
            select = (features['operation'] == '<=').to_numpy()
            features.loc[select, 'value'] = thresholds[select]

        # Drop duplicate features.
        features.drop_duplicates(inplace=True)

        # Create rule pairs for each feature
        temp = features.copy(True)
        temp['operation'].replace({'<=': '>', '==': '!=', '': 'not'}, inplace=True)
        features = pd.concat((features, temp))

        # Create/sort multi-index that will be used during transformation.
        self.features = pd.MultiIndex.from_frame(features)
        self.features = self.features.sortlevel((0, 1, 2))[0]

        # Update respective self attributes based on the selected features.
        if '==' in self.features.levels[1]:
            names = self.features.get_loc_level('==', 'operation')[1].get_level_values('feature').unique()
            self.enc = {k: self.enc[k] for k in names}
        else:
            self.enc = {}

        if '' in self.features.levels[1]:
            names = self.features.get_loc_level('', 'operation')[1].get_level_values('feature').unique()
            self.maps = {k: self.maps[k] for k in names}
        else:
            self.maps = {}

        if '<=' in self.features.levels[1]:
            names = self.features.get_loc_level('<=', 'operation')[1].get_level_values('feature').unique()
            self.thresh = \
                {k: self.features
                    .get_loc_level([k, '<='], ['feature', 'operation'])[1]
                    .get_level_values('value')
                    .to_numpy(dtype=float)
                 for k in names}
            self.ordinal = names.to_list()
            if self.returnOrd:
                self.scaler = StandardScaler().fit(X[names])
        else:
            self.thresh = {}
            self.ordinal = []
            self.scaler = None

        return self

    def transform(self, X: DataFrame) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        """Binarize features. Binary features are sorted name-operation-value.
        Args:
            X (DataFrame): Original features
        Returns:
            A (DataFrame): Binarized features with MultiIndex column labels
            Xstd (DataFrame, optional): Standardized ordinal features
        """

        result = DataFrame(
            np.zeros((X.shape[0], len(self.features)), dtype=int),  # Type consistent with original FeatureBinarizer
            columns=self.features,
            index=X.index
        )

        # Using to_numpy() speeds up the overall transform by more than 2x because indices aren't created/aligned.
        # Futhermore, using numpy raises warnings for NaN/None values which is what we want here.
        for feature, test, value in result.columns:
            if test == '<=':
                v = X[feature].to_numpy() <= value
            elif test == '>':
                v = X[feature].to_numpy() > value
            elif test == '==':
                v = X[feature].to_numpy() == value
            elif test == '!=':
                v = X[feature].to_numpy() != value
            elif test == '':
                v = X[feature].to_numpy() == self.maps[feature].index[1]
            elif test == 'not':
                v = X[feature].to_numpy() == self.maps[feature].index[0]
            else:
                raise RuntimeError('Test operation \'{}\' not supported.' % test)

            # Faster to replace column with numpy because there is no index to align/join
            result[(feature, test, value)] = v.astype(int)

        if self.threshStr:
            result.columns.set_levels(result.columns.levels[2].astype(str), 'value', inplace=True)

        # This is taken from FeatureBinarizer.
        if self.returnOrd:
            # Standardize ordinal features
            Xstd = self.scaler.transform(X[self.ordinal])
            Xstd = DataFrame(Xstd, index=X.index, columns=self.ordinal)
            return result, Xstd
        else:
            return result