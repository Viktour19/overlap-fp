import sys, os
folder = '/home/victora/PositivityViolation/'

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils import write_model
from comp_preprocessing import get_data
from comp_overrule import overrulefit

from multiprocessing.pool import Pool
from multiprocessing import cpu_count

DATA_PATH = folder + 'data/fp_select.csv'

def model(data_path = DATA_PATH, encode=True):
    
    X_df, a, y = get_data(data_path, encode=encode)
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]
    
    base_estimator = LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced", random_state=2, solver='lbfgs')
    param_grid = {'C': np.logspace(-5, 0, 20)}
    search = GridSearchCV(base_estimator, param_grid, cv=5, scoring='roc_auc')

    model = make_pipeline(StandardScaler(), search)
    model.fit(X_df, a)
    
    clrmodel = make_pipeline(StandardScaler(), model.steps[1][1].best_estimator_) 
    clrmodel.fit(X_df, a)

    return clrmodel, X_df, a


def learn_orules(LAMBDA0_s=0.1, LAMBDA1_s=0.1, logspace=10, data_path = DATA_PATH, encode=True, cmodel=None):
    
    if cmodel is None:
        clrmodel, X_df, a = model(data_path)
        write_model(clrmodel, 'clrmodel')
    else:
        X_df, a, y = get_data(data_path)
        
        X_df = X_df[~y.isna()]
        a = a[~y.isna()]
        y = y[~y.isna()]
        
        clrmodel = cmodel
    
    LAMBDA_0 = np.logspace(-7, -0.1, logspace)
    LAMBDA_1 = np.logspace(-7, -0.1, logspace)
    
    RS_s = None
    results = []
    
    pool = Pool(processes=cpu_count() - 1)
    for lambda_0 in tqdm(LAMBDA_0):
        for lambda_1 in tqdm(LAMBDA_1):
            results.append(pool.apply_async(overrulefit, (X_df, a), \
                                            {'LAMBDA0_s': LAMBDA0_s, 'LAMBDA1_s': LAMBDA1_s, \
                                             'model':clrmodel, 'LAMBDA0_o': lambda_0, 'LAMBDA1_o': lambda_1, 'RS_s': RS_s}))
            
            
    pool.close()
    pool.join()
    results_data = []
    for res in results:
        try:
            results_data.append(res.get())
        except AssertionError as e:
            continue
    
    results = []
    
    for M, RS_s, RS_o, auc, score_base in results_data:
        if RS_o is None:
            results.append([RS_s.complexity(), (0, 0), auc, score_base, lambda_0, lambda_1])
        else:
            results.append([RS_s.complexity(), RS_o.complexity(), auc, score_base, lambda_0, lambda_1])
        
    return results
