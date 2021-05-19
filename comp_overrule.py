import sys, os
folder = '/home/victora/PositivityViolation/'
sys.path.append(folder + 'overrule')

from overrule.ruleset import BCSRulesetEstimator
from overrule.overrule import OverRule2Stage
from overrule.baselines import propscore

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

from comp_preprocessing import get_data
from utils import read_model, rule_str

from multiprocessing.pool import Pool
from multiprocessing import cpu_count

import time 

N_REF_MULT_s=0.3
N_REF_MULT_o=0

ALPHA_s=0.95
ALPHA_o=0.90

D=10  # Maximum extra rules per beam seach iteration
K=10  # Maximum results returned during beam search
B=28  # Width of Beam Search

SEED=0
CNF=False
CAT_COLS = []

np.random.seed(SEED)

DATA_PATH = folder + 'data/fp_select.csv'

def overrulefit(X_df_sample, a_sample, LAMBDA0_s=None, LAMBDA1_s=None, model=None, LAMBDA0_o=None, LAMBDA1_o=None, RS_s = None, only_support=False):
    
    O = propscore.PropensityOverlapEstimator(estimator=model)
    
    if RS_s is None:
        RS_s = BCSRulesetEstimator(n_ref_multiplier=N_REF_MULT_s, alpha=ALPHA_s, lambda0=LAMBDA0_s, lambda1=LAMBDA1_s, B=B, CNF=CNF, 
                                   cat_cols=CAT_COLS, seed=SEED, K=K, D=D, binarizer='tree')
        RS_s.fit(X_df_sample, a_sample)
    
    if only_support:
            TPR = RS_s.predict(X_df_sample).mean()
            FPR = RS_s.relative_volume
            auc = 1/2 -  (FPR)/2 + TPR/2
            return None, RS_s, None, auc, None
        
    RS_o = BCSRulesetEstimator(n_ref_multiplier=N_REF_MULT_o, alpha=ALPHA_o, lambda0=LAMBDA0_o, lambda1=LAMBDA1_o, B=B, CNF=CNF, 
                               cat_cols=CAT_COLS, seed=SEED, binarizer='default')

    
    try:
        M = OverRule2Stage(O, RS_o, RS_s, refit_s=False)
        M.fit(X_df_sample, a_sample)

        TPR = RS_s.predict(X_df_sample).mean()
        FPR = RS_s.relative_volume
        auc = 1/2 -  (FPR)/2 + TPR/2
        score_base = M.score_vs_base(X_df_sample)

        return M, RS_s, RS_o, auc, score_base
    
    except AssertionError as e:
        
        TPR = RS_s.predict(X_df_sample).mean()
        FPR = RS_s.relative_volume
        auc = 1/2 -  (FPR)/2 + TPR/2
        return None, RS_s, None, auc, None

                
def learn_srules(logspace=10, data_path = DATA_PATH):
    
    X_df, a, y = get_data(data_path)
    
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]

    LAMBDA_0 = np.logspace(-7, -0.1, logspace)
    LAMBDA_1 = np.logspace(-7, -0.1, logspace)
    
    results = []
    pool = Pool(processes=cpu_count() - 1)
    
    for lambda_0 in tqdm(LAMBDA_0):
        for lambda_1 in tqdm(LAMBDA_1):
            results.append(pool.apply_async(overrulefit, (X_df, a), \
                                            {'LAMBDA0_s': lambda_0, 'LAMBDA1_s': lambda_1, 'only_support':True}))
    
    pool.close()
    pool.join()
    
    results_data = [res.get() for res in results]
    results = []
    for M, RS_s, RS_o, auc, score_base in results_data:
        results.append([RS_s.complexity(), None, auc, None, lambda_0, lambda_1])
        
    return results


def learn_s_orules(model_path, LAMBDA0_s, LAMBDA1_s, LAMBDA0_o, LAMBDA1_o, data_path = DATA_PATH, encode=True, only_support=False):
    
    model = read_model(model_path)
    X_df, a, y = get_data(data_path, encode=encode)
    
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]
    
    M, RS_s, RS_o, auc, score_base = overrulefit(X_df, a, LAMBDA0_s=LAMBDA0_s, LAMBDA1_s=LAMBDA1_s, LAMBDA0_o=LAMBDA0_o, LAMBDA1_o=LAMBDA1_o, model=model, only_support=only_support)

    return M, RS_s, RS_o, auc, score_base

def plt_cl_lit(results, rtype="support", title=None):
    
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)

    complexities_c = []
    complexities_l = []
    accuracy = []
    
    j = 0
    if rtype == 'overlap':
        j = 1
    
    for result in results:
        complexities_c.append(result[0 +j][0])
        complexities_l.append(result[0 +j][1])
        accuracy.append(result[2 +j])

    sns.lineplot(complexities_c, complexities_l, ax=axs)
    axs.set_xlabel('Number of Clauses')
    axs.set_ylabel('Number of Literals')
    title = "{} estimator".format(rtype)
    axs.set_title(title)
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    fig.savefig(folder + 'figures/' + rtype + 'clause' + timestamp + '.pdf')
    
    return axs


def get_sem_optim(results, rtype="support"):
    
    j = 0
    if rtype == 'overlap':
        j = 1
        
    complexities_c = []
    accuracy = []
    for result in results:
        complexities_c.append(result[0 +j][0])
        accuracy.append(result[2 +j])
        
    data = pd.DataFrame({'accuracy': accuracy}, index = complexities_c)
    select_x = data[(data['accuracy'] >= (np.max(accuracy) - (stats.sem(accuracy))))].index.min()
#     select_x = data[(data['accuracy'] >= .90)].index.min()    
    
    
    newmax = 0
    max_result = None
    for result in results:
        if result[0+j][0] == select_x:
            rmax = result[2+j]
            if rmax > newmax:
                newmax = rmax
                max_result = result
    print(np.max(accuracy))         
    return max_result

def plt_sem(results, rtype="support"):
    
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    j = 0
    if rtype == 'overlap':
        j = 1
        
    complexities_c = []
    accuracy = []
    for result in results:
        complexities_c.append(result[0 +j][0])
        accuracy.append(result[2 +j])
        
    data = pd.DataFrame({'accuracy': accuracy}, index = complexities_c)
    
    sns.lineplot(complexities_c, accuracy, ax=axs)
    
    select_x = data[(data['accuracy'] >= (np.max(accuracy) - stats.sem(accuracy)))].index.min()
    max_x = data[data['accuracy'] == np.max(accuracy)].index.min()

    axs.axvline(x=select_x, linestyle='--', label='min_x s.t. y > max - sem', color='black')
    axs.axvline(x=max_x, linestyle='--', label='max', color='brown')

    axs.set_xlabel('Number of Clauses')
    axs.set_ylabel('Balanced accuracy')
    title = "{} estimator".format(rtype)
    axs.set_title(title)
    plt.tight_layout()
    axs.legend()
    
    timestamp = str(int(time.time()))
    fig.savefig(folder + 'figures/' + rtype + 'accuracy' + timestamp + '.pdf')
    
    return fig, axs
    

    
def get_overlap_violations(RS_s, RS_o, data_path = DATA_PATH, encode=True):
    
    X_df, _, _ = get_data(data_path, encode=encode)
    
    support_set = X_df[eval(rule_str(RS_s.rules()))]
    overlap_set = X_df[eval(rule_str(RS_o.rules()))]

    intersection = set(list(support_set.index)).intersection(set(list(overlap_set.index)))
    violating_index = list(set(X_df.index) - intersection)
    overlap_index = list(intersection)

    return overlap_index, violating_index