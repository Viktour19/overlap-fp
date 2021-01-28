folder = '/home/victora/PositivityViolation/'

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

import numpy as np
import pandas as pd
import time

encoding = pd.read_csv(folder + 'data/encoding.csv')

def get_lbl(var):
    if var in list(encoding['var_name']):
        return encoding[encoding['var_name'] == var].iloc[0].label

def select_data(data_path=folder + 'data/KE_2014_Preprocessed.csv'):
    data = pd.read_csv(data_path)
    vvars = list(encoding[encoding['selected'] == 1].var_name)
    data = data[vvars + ['outcome', 'treatment']]
    
    timestamp = str(int(time.time()))
    data.to_csv(folder + 'data/fp_select' + timestamp + '.csv', index=False)
    return data

    
def get_data(data_path=folder + 'data/fp_select.csv', cols=None, encoding_path=folder + 'data/encoding.csv', encode=True):
    
    data = pd.read_csv(data_path)
    encoding = pd.read_csv(encoding_path)
    
    encoding_dict = {'O': list(encoding[encoding['encoding'] == 'O'].var_name),
                     'L': list(encoding[encoding['encoding'] == 'L'].var_name),
                     'N': list(encoding[encoding['encoding'] == 'N'].var_name)}
    
    
    if cols is None:
        X = data[data.columns[:-2]]
    else:
        X = data[cols]
    
    X = X.apply(lambda x: x.fillna(x.median()),axis='rows')
    y = data['outcome'] * 1
    a = data['treatment'] * 1
    
    if not encode:
        return X, a, y

    # Select and Encode ordinal features
    enc = OrdinalEncoder()
    ord_features = encoding_dict['O']
    ord_features = list(set(ord_features).intersection(set(X.columns)))
    ord_features.sort()
    ord_data = enc.fit_transform(X[ord_features])

    # Select and Encode nominal features
    enc = OneHotEncoder(categories='auto')
    nom_features_ = encoding_dict['L']
    nom_features_ = list(set(nom_features_).intersection(set(X.columns)))
    nom_features_.sort()
    nom_data = enc.fit_transform(X[nom_features_].astype(int))
    nom_features = enc.get_feature_names(nom_features_)

    # Select the discrete features
    dis_features = encoding_dict['N']
    dis_features = list(set(dis_features).intersection(set(X.columns)))
    dis_features.sort()
    index_cols = [col for col in X.columns if '_index' in col]
    
    dis_features = dis_features + index_cols 
    
    dis_data = X[dis_features].values

    # Combine all the features
    X = np.concatenate((ord_data, nom_data.toarray(), dis_data), axis=1)
    features_names = np.concatenate((ord_features, nom_features, dis_features))

    X_df = pd.DataFrame(X, columns=features_names, index=data.index)
    encoded = pd.concat((X_df, data[['outcome', 'treatment']]), axis=1)

    return X_df, a, y


def get_varencoding():
    
    with open(folder + 'data/varencoding.txt', 'r') as f: 
        entire_doc = f.read()

    vencoding = entire_doc.split(';')

    var_encoding = dict()
    for var in vencoding:
        var = var.strip()
        splits = var.split('\n')
        k = splits[0].replace('define', '').strip()
        vs = {}
        for i in range(1, len(splits)):
            v = splits[i].strip().split(" \"")
            vs.update({v[0]: v[1].replace('"', '')})

        var_encoding.update({k.lower(): vs})
    
    return var_encoding