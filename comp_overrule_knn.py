import sys, os
folder = '/home/victora/PositivityViolation/'

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
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

def model(data_path = DATA_PATH):
    
    X_df, a, y = get_data(data_path)
    
    base_estimator = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(np.arange(8, 50, 2)), 'metric': ['minkowski']}
    search = GridSearchCV(base_estimator, param_grid, cv=5, scoring='roc_auc')

    model = make_pipeline(StandardScaler(), search)
    model.fit(X_df, a)
    
    knnmodel = make_pipeline(StandardScaler(), model.steps[1][1].best_estimator_) 
    knnmodel.fit(X_df, a)

    return knnmodel, X_df, a


def learn_orules(LAMBDA0_s=0.1, LAMBDA1_s=0.1, logspace=10, data_path = DATA_PATH, kmodel=None):
    
    if kmodel is None:
        knnmodel, X_df, a = model(data_path)
        write_model(knnmodel, 'knnmodel')
    else:
        X_df, a, _ = get_data(data_path)
        knnmodel = kmodel
    
    LAMBDA_0 = np.logspace(-7, -0.1, logspace)
    LAMBDA_1 = np.logspace(-7, -0.1, logspace)
    
    RS_s = None
    results = []
    
    pool = Pool(processes=cpu_count() - 1)
    for lambda_0 in tqdm(LAMBDA_0):
        for lambda_1 in tqdm(LAMBDA_1):
            results.append(pool.apply_async(overrulefit, (X_df, a), \
                                            {'LAMBDA0_s': LAMBDA0_s, 'LAMBDA1_s': LAMBDA1_s, \
                                             'model':knnmodel, 'LAMBDA0_o': lambda_0, 'LAMBDA1_o': lambda_1, 'RS_s': RS_s}))
            
            
    pool.close()
    pool.join()
    
    results_data = [res.get() for res in results]
    results = []
    for M, RS_s, RS_o, auc, score_base in results_data:
        if RS_o is None:
            results.append([RS_s.complexity(), (0, 0), auc, score_base, lambda_0, lambda_1])
        else:
            results.append([RS_s.complexity(), RS_o.complexity(), auc, score_base, lambda_0, lambda_1])
        
    return results
