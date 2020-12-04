from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import sys
# sys.path.append('./overlap-code')
# from exps.supp_synthetic.synth_utils import compliance

def get_data(data=None, cols=None):
    
    if data is None:
        data = pd.read_csv('/home/victora/PositivityViolation/data/fp_injectables_data.csv')

    encoding_dict = {'O': ['v106', 'v133', 'v149', 'v157', 'v158', 'v159', 'v190', 'v301'],
                     'L': ['v101', 'v102', 'v113', 'v116', 'v119', 'v120', 'v121',
                           'v122', 'v123', 'v124', 'v125', 'v127', 'v128', 'v129', 'v130',
                           'v131', 'v135', 'v139', 'v140', 'v150', 'v151', 'v153', 'v155',
                           'v161', 'v213', 'v217', 'v228', 'v312', 'v313', 'v361', 'v364',
                           'v393', 'v394', 'v501', 'v502', 'v513', 'v532', 'v536', 'v602',
                           'v605', 'v623', 'v624', 'v625', 'v626', 'v170', 'v216', 'v244',
                           'v379',  'v380' ],
                     'N': ['v136', 'v137', 'v138', 'v152', 'v167', 'v191', 'v201', 'v202',
                           'v203', 'v204', 'v205', 'v206', 'v207', 'v208', 'v209', 'v210',
                           'v215', 'v218', 'v219', 'v220', 'v224', 'v226', 'v227', 'v235',
                           'v238', 'v525', 'v531', 'v613', 'v614', 'v627', 'v628', 'v629', 
                           'v115', 'v104']}
    if cols is None:
        remove = ['v213', 'v210']
        X_ = data[data.columns[:-2]]
        X = X_.drop(columns=remove)
        X = X.apply(lambda x: x.fillna(x.median()),axis='rows')
        c = X.corr().abs()
        c = c * np.tri(c.shape[0], c.shape[1], -1)
        c = c.transpose()

        corr_cols = [col for col in c.columns if any(c[col] > .96)]
        X.drop(columns=corr_cols, inplace=True)
    else:
        X = data[data.columns[:-2]]
        X = X[cols]
        X = X.apply(lambda x: x.fillna(x.median()),axis='rows')

    y = data['outcome'] * 1
    a = data['treatment'] * 1

    # Select and Encode ordinal features
    enc = OrdinalEncoder()
    ord_features = encoding_dict['O']
    ord_features = list(set(ord_features).intersection(set(X.columns)))
    ord_data = enc.fit_transform(X[ord_features])

    # Select and Encode nominal features
    enc = OneHotEncoder(categories='auto')
    nom_features_ = encoding_dict['L']
    nom_features_ = list(set(nom_features_).intersection(set(X.columns)))
    nom_data = enc.fit_transform(X[nom_features_].astype(int))
    nom_features = enc.get_feature_names(nom_features_)

    # Select the discrete features
    dis_features = encoding_dict['N']
    dis_features = list(set(dis_features).intersection(set(X.columns)))
    index_cols = [col for col in X.columns if '_index' in col]
    
    dis_features = dis_features + index_cols 
    
    dis_data = X[dis_features].values

    # Combine all the features
    X = np.concatenate((ord_data, nom_data.toarray(), dis_data), axis=1)
    features_names = np.concatenate((ord_features, nom_features, dis_features))

    X_df = pd.DataFrame(X, columns=features_names, index=data.index)
    encoded = pd.concat((X_df, data[['outcome', 'treatment']]), axis=1)

    return X_df, a, y



def fatom(f, o, v, fmt='%.3f'):
    if o in ['<=', '>', '>=', '<', '==']:
        if isinstance(v, str):
            return ('(X_df["%s"] %s %s)') % (f,o,v)
        else:
            return ('(X_df["%s"] %s '+fmt+')') % (f,o,v)
    elif o == 'not':
        return '~X_df["%s"].astype(bool)' % f
    else:
        return 'X_df["%s"].astype(bool)' % f

def rule_str(C, fmt='%.3f'):
    s = '  '+'| '.join(['(%s)' % (' & '.join([fatom(a[0], a[1], a[2], fmt=fmt) for a in c])) for c in C])
    return s


# def rules_stats(r_rules, df):

#     rules = r_rules(transform=lambda a,b: b, fmt='%.1f')
#     n_rules = float(len(rules))
#     n_rules_literals = float(np.sum([len(rule) for rule in rules]))

#     # Record more detailed rules information, e.g., proportion covered
#     D = pd.concat([
#         df,
#         pd.DataFrame(np.ones_like(a_sample), columns=['support_set'])
#         ], axis=1)
#     Cs = compliance(D, rules)

#     # This is everywhere, to be clear
#     I1 = np.where(D['support_set'].values==1)[0]

#     rule_stats = []
#     for i in range(len(rules)):
#         # Instances covered by rule
#         d = {}
#         d['rule'] = rules[i]
#         d['n_covered'] = float(Cs[i][:,I1].prod(0).sum())
#         d['p_covered'] = float(Cs[i][:,I1].prod(0).mean())
#         rule_stats.append(d)
#     return rule_stats