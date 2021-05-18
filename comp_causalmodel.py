from causallib.estimation import IPW, MarginalOutcomeEstimator, StratifiedStandardization
from causallib.evaluation import PropensityEvaluator, OutcomeEvaluator
from causallib.estimation import DoublyRobustVanilla

from sklearn.linear_model import LogisticRegression, PoissonRegressor, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.metrics import make_scorer
from sklearn.utils import resample
from causallib.utils.stat_utils import robust_lookup

from utils import write_model
from comp_preprocessing import get_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import time
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

folder = '/home/victora/PositivityViolation/'

def weighted_auc_scorer(clf, X, a_true):
    
    a_proba = pd.DataFrame(clf.predict_proba(X))
    weight_matrix = a_proba.rdiv(1.0)
    
    a_true = a_true.reset_index(drop=True)
    prevalence = a_true.value_counts(normalize=True, sort=False)
    prevalence_per_subject = a_true.replace(prevalence)  

    weight_matrix = weight_matrix.multiply(prevalence_per_subject, axis="index")
    weights = robust_lookup(weight_matrix, a_true)
    
    score =  roc_auc_score(a_true, clf.predict_proba(X)[:, 1], sample_weight=weights)
    return 1 - abs(score - 0.5)


def basemodel(data_path = folder + 'data/fp_select.csv', encode=True, method='sigmoid', scoring=weighted_auc_scorer):
    X_df, a, y = get_data(data_path, encode=encode)
    
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]

    strartify_by = (a*2) + y
    X_train, X_test, a_train, a_test, y_train, y_test = train_test_split(X_df, a, y, train_size=0.7, test_size=0.3, shuffle=True, \
                                                                         random_state=1, stratify=strartify_by)
    
    base_estimator = LogisticRegression(penalty="l2", max_iter=3000, class_weight="balanced", random_state=2, solver='lbfgs')
    learner = CalibratedClassifierCV(base_estimator=base_estimator, cv=3, method=method)
    param_grid = {'base_estimator__C': np.logspace(-2, 0, 20)}
    search = GridSearchCV(learner, param_grid, cv=3, scoring=scoring)

    return search, X_train, X_test, a_train, a_test, y_train, y_test



def propensity_model(data_path = folder + 'data/fp_select.csv', encode=True, method='sigmoid'):
    
    search, X_train, X_test, a_train, a_test, y_train, y_test = basemodel(data_path, encode, method)

    ipw = IPW(make_pipeline(StandardScaler(), search), use_stabilized=True)
    ipw.fit(X_train, a_train)
    
#     ipw = IPW(make_pipeline(StandardScaler(), ipw.learner.steps[1][1].best_estimator_), use_stabilized=True)
#     ipw.fit(X_train, a_train)
    
    evaluations = causal_eval(ipw,  X_test, a_test, y_test)
    return evaluations, X_test, a_test, y_test

def outcome_model(data_path = folder + 'data/fp_select.csv', encode=True, scoring='roc_auc'):

    X_df, a, y = get_data(data_path, encode=encode)
    
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]

    strartify_by = (a*2) + y
    X_train, X_test, a_train, a_test, y_train, y_test = train_test_split(X_df, a, y, train_size=0.7, test_size=0.3, shuffle=True, \
                                                                         random_state=1, stratify=strartify_by)
    
    base_estimator = LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced", random_state=2, solver='liblinear')
    learner = CalibratedClassifierCV(base_estimator=base_estimator, cv=3, method='sigmoid')
    std = StratifiedStandardization(make_pipeline(StandardScaler(), learner))

    std.fit(X_train, a_train, y_train)
    plots = ["common_support"]
    
    evaluator = OutcomeEvaluator(std)
    evaluations = evaluator.evaluate_simple(X_test.astype(float), a_test, y_test, evaluator._numerical_classification_metrics, plots)

    fig = evaluations.plots['common_support'].get_figure()
    fig.set_size_inches(4, 4) 
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    
#     fig.savefig(folder + 'figures/causaleval' + timestamp + '.pdf')
    
    return evaluations, X_test, a_test, y_test

def dr_model():
    # dr = DoublyRobustVanilla(std, ipw)
    # dr.fit(X_train[~y_train.isna()], a_train[~y_train.isna()], y_train[~y_train.isna()], refit_weight_model=False)

    pass

def causal_eval(model, X_test, a_test, y_test):
    
    plots=["roc_curve", "covariate_balance_slope", "weight_distribution", "calibration"]
    evaluator = PropensityEvaluator(model)
    evaluations = evaluator.evaluate_simple(X_test.astype(float), a_test, y_test, plots=plots)

    fig = evaluations.plots['covariate_balance_slope'].get_figure()
    fig.set_size_inches(8, 8) 
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    fig.savefig(folder + 'figures/causaleval' + timestamp + '.pdf')
    
    return evaluations

def bootstrap_marginal(data_path = folder + 'data/fp_select.csv', n_bootstrap = 1000, title="distribution of marginal diff", effect_type='or'):
    
    X_df, a, y = get_data(data_path)
    X_df = X_df[~y.isna()]
    a = a[~y.isna()]
    y = y[~y.isna()]

    strartify_by = (a*2) + y
    X_train, X_test, a_train, a_test, y_train, y_test = train_test_split(X_df, a, y, train_size=0.7, test_size=0.3, shuffle=True, \
                                                                         random_state=1, stratify=strartify_by)
    moe = MarginalOutcomeEstimator(None).fit(X_train, a_train, y_train)
    outcomes = []
    for i in tqdm(range(n_bootstrap)):
        X_r, a_r, y_r = resample(X_test, a_test, y_test, n_samples=None, random_state=i)
        X_r = X_r.reset_index(drop=True)
        a_r = a_r.reset_index(drop=True)
        y_r = y_r.reset_index(drop=True)
        
        outcome = moe.estimate_population_outcome(X_r, a_r, y_r)
        outcomes.append(moe.estimate_effect(outcome[1], outcome[0], effect_types=effect_type)[effect_type])
        
    plt.hist(outcomes)
    plt.title(title)
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    
    plt.savefig(folder + 'figures/outcomes' + timestamp + '.pdf')

    median = np.median(outcomes)
    lower = np.percentile(outcomes, 2.5)
    upper = np.percentile(outcomes, 97.5)
    
    return median, lower, upper
    
def bootstrap_effects(ipw, X_test, a_test, y_test, n_bootstrap = 1000, title="distribution of ATE", effect_type='or'):

    effects = []
    for i in tqdm(range(n_bootstrap)):
        X_r, a_r, y_r = resample(X_test, a_test, y_test, n_samples=None, random_state=i)

        X_r = X_r.reset_index(drop=True)
        a_r = a_r.reset_index(drop=True)
        y_r = y_r.reset_index(drop=True)

        potential_outcomes = ipw.estimate_population_outcome(X_r, a_r, y_r)
        causal_effect = ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0], effect_types=effect_type)[effect_type]
        effects.append(causal_effect)
        
    plt.hist(effects)
    plt.title(title)
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    plt.savefig(folder + 'figures/effects' + timestamp + '.pdf')

    median = np.nanmedian(effects)
    lower = np.nanpercentile(effects, 2.5)
    upper = np.nanpercentile(effects, 97.5)
    
    return median, lower, upper


def placebo_effects(basemodel, X_test, a_test, y_test, n_bootstrap=1000, title="distribution of placebo"):
    
    ipw = IPW(make_pipeline(StandardScaler(), basemodel), use_stabilized=True)
    
    placebo_effects = []
    p_a = a_test.mean()
    for i in tqdm(range(n_bootstrap)):

        X_boots, a_boots, y_boots = resample(X_test, a_test, y_test, n_samples=None, random_state=i)
        random_a = np.random.binomial(1, p_a, size=X_boots.shape[0])

        random_a = pd.Series(random_a)
        X_boots = X_boots.reset_index(drop=True)
        y_boots = y_boots.reset_index(drop=True)

        ipw.fit(X_boots, random_a)
        potential_outcomes = ipw.estimate_population_outcome(X_boots, random_a, y_boots)
        placebo_effects.append(ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0], effect_types="or")['or'])
    
    plt.hist(placebo_effects)
    plt.title(title)
    
    timestamp = str(int(time.time()))
    plt.tight_layout()
    plt.savefig(folder + 'figures/placeboeffects' + timestamp + '.pdf')
    
    return np.mean(placebo_effects)