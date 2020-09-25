import sys, os
import numpy as np
sys.path.append('./overlap-code')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, datasets
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from overrule.overrule import OverRule2Stage
from overrule.baselines import knn, marginal, propscore, svm
from overrule.support import SVMSupportEstimator, SupportEstimator
from overrule.overlap import SupportOverlapEstimator
from overrule.ruleset import BCSRulesetEstimator, RulesetEstimator


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv('./data/fp_injectables_data.csv')

y = df['outcome'] * 1
a = df['treatment'] * 1
X = df[df.columns[:-2]]
X = X.apply(lambda x: x.fillna(x.median()),axis='rows')

encoding = pd.read_csv('./data/encoding.csv')

# Select and Encode ordinal features
v = encoding[encoding['encoding'] == 'O']['var_name'].values
enc = OrdinalEncoder()
ord_data = enc.fit_transform(X[v])
ord_features = v

# Select the discrete features
v = encoding[encoding['encoding'] == 'N']['var_name'].values
dis_data = X[v].values
dis_features = v

# Select and Encode nominal features
v = encoding[encoding['encoding'] == 'L']['var_name'].values
j = X[v].astype(int)
enc = OneHotEncoder(categories='auto')
nom_data = enc.fit_transform(j.astype(int))
nom_features = enc.get_feature_names(v)


# Combine all the features
X_arr = np.concatenate((ord_data, j, dis_data), axis=1)
features_names = np.concatenate((ord_features, v, dis_features))
# X_arr = np.concatenate((ord_data, nom_data.toarray(), dis_data), axis=1)
# features_names = np.concatenate((ord_features, nom_features, dis_features))

print(X_arr.shape)
X_df = pd.DataFrame(X_arr, columns=features_names)


SEED=0
TUNE_XI_SUPPORT=True
COVERAGE=True
CNF=True
CNFSynth=True
VERBOSE=True
FIX_MARGINAL=False

ALPHA=0.98
N_REF_MULT=1
LAMBDA0=1e-5
LAMBDA1=1e-3

D=20  # Maximum extra rules per beam seach iteration
K=20  # Maximum results returned during beam search
B=300  # Width of Beam Search

nr = 2  # Rare Features
rare_prob = 0.01  # Rare Feature Probability
ni = 2   # Interaction Features
nn = 6 # Normal (i.e., prob 0.5) Features
nd = nn + ni + nr  # Total Features

np.random.seed(SEED)
w_eps = 1e-8
cat_cols = list(v)

RS_s = BCSRulesetEstimator(n_ref_multiplier=N_REF_MULT, alpha=ALPHA, lambda0=LAMBDA0, lambda1=LAMBDA1, B=B, cat_cols=cat_cols)
RS_s.fit(X_df, a)

print('Number of reference samples: {}'.format(RS_s.refSamples.shape[0]))
print('Coverage of data points: %.3f, Requested >= %.3f' % (RS_s.predict(X_df).mean(), RS_s.M.alpha))
print('Coverage of reference points: %.3f' % RS_s.predict(RS_s.refSamples).mean())

outfile = open('support_metrics.txt', 'a+')

outfile.write('Params: n_ref_multiplier {}, alpha {}, lambda0 {}, lambda1 {}, B {}\n'.format(N_REF_MULT, ALPHA, LAMBDA0, LAMBDA1, B))
outfile.write('Number of reference samples: {}\n'.format(RS_s.refSamples.shape[0]))
outfile.write('Coverage of data points: %.3f, Requested >= %.3f\n' % (RS_s.predict(X_df).mean(), RS_s.M.alpha))
outfile.write('Coverage of reference points: %.3f\n' % RS_s.predict(RS_s.refSamples).mean())

outfile.close()