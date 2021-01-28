import sys, os
folder = '/home/victora/PositivityViolation/'
sys.path.append(folder + 'overrule')

from exps.supp_synthetic.synth_utils import compliance
from comp_preprocessing import get_lbl, get_varencoding

import numpy as np
import pandas as pd
import time
import pickle
import glob

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


def rules_stats(r_rules, df, a):

    rules = r_rules(transform=lambda a,b: b, fmt='%.1f')
    n_rules = float(len(rules))
    n_rules_literals = float(np.sum([len(rule) for rule in rules]))

    # Record more detailed rules information, e.g., proportion covered
    D = pd.concat([df, pd.DataFrame(np.ones_like(a), columns=['support_set'])], axis=1)
    Cs = compliance(D, rules)

    # This is everywhere, to be clear
    I1 = np.where(D['support_set'].values==1)[0]

    rule_stats = []
    for i in range(len(rules)):
        # Instances covered by rule
        d = {}
        d['rule'] = rules[i]
        d['n_covered'] = float(Cs[i][:,I1].prod(0).sum())
        d['p_covered'] = float(Cs[i][:,I1].prod(0).mean())
        rule_stats.append(d)
    
    return rule_stats


def transcribe(rule_stats):
    
    var_encoding = get_varencoding()
    clauses = []
    
    for single_rset in rule_stats:
        single_rset_rule = single_rset['rule']
        clause = []
        for i in range(len(single_rset_rule)):
            rule = single_rset_rule[i]
            
            if 'index' in rule[0]:
                var = rule[0]
                rule_str = var + " " + rule[1] + " " +  str(round(rule[2]))

            elif '_' in rule[0]:
                var = rule[0].split("_")[0]
                level = rule[0].split("_")[1]

                var_str = get_lbl(var)
                level_str = var_encoding[var][level]

                rule_str = var_str + " is " + rule[1] + " \"" + level_str + "\""
                
            else:
                var = rule[0]
                var_str = get_lbl(var)
                rule_str = var_str + " " + rule[1] + " " +  str(rule[2])
                           
                
            if i != len(single_rset_rule) - 1:
                rule_str = rule_str + " and"
            
            clause.append(rule_str)
        clauses.append(clause)
        
    return clauses


def write_model(model, name):
    timestamp = str(int(time.time()))
    obj = {"clf": model}
    
    with open('pickle/'+ name + '_' + timestamp + '.pickle', 'wb') as f_out:
        
        pickle.dump(obj, f_out)
        f_out.close()
        
def read_model(modelpath):
    with open(modelpath, 'rb') as fp:
        model_pkl = pickle.load(fp)
        fp.close()

    model = model_pkl['clf']
    return model


def recent_model(name):
    
    files = []
    times = []
    for file_name in glob.glob('pickle/' + name +'*.pickle'):
        files.append(file_name)
        times.append(int(file_name.split('_')[1].split('.')[0]))
    
    if len(times) > 0:
        recent_model = files[np.argsort(times)[-1]]
        return recent_model