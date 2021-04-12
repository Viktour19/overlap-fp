import sys, os
folder = '/home/victora/PositivityViolation/'
sys.path.append(folder + 'positivitree')

from positivitree import PositiviTree
from comp_preprocessing import get_lbl, get_data, get_varencoding
from positivitree.tree_vis import visualize_leaves_static
import time

DATA_PATH = folder + 'data/fp_select.csv'


def transcribe(violating_leaves, var_encoding=None):
    var_encoding = get_varencoding(path=var_encoding)
    rules = []
    
    for leaf in violating_leaves:
        leaves_rules = []
        for query in leaf['query']:
            
            violation = [query.feature_name, query.threshold_sign, query.threshold_value]
            
            level  = ''
            s = violation[1]
            t = violation[2]
            
            if '_' in violation[0]:
                
                v, l = violation[0].split('_')
#                 print(v, l, var_encoding[v], get_lbl(v))
                var = get_lbl(v)
                level = var_encoding[v][l]
                
                if violation[2] == 0.5:
                    if violation[1] == '<' or violation[1] == '<=':
                        t = 0
                        s = '='
                    if violation[1] == '>' or violation[1] == '>=':
                        t = 1
                        s = '='
                
            else:
                var = get_lbl(violation[0])
                
            leaves_rules.append((var, level, s, t))
        
        rules.append(leaves_rules)
        
    return rules
        
        
def learn_rules(data_path=DATA_PATH, var_encoding=None, relative=False):
    
    X_df, a, y = get_data(data_path)
    
    ptree = PositiviTree(X_df, a, violation_cutoff=0.1, consistency_cutoff=0.6, n_consistency_tests=200, relative_violations=relative, \
                         dtc_kws={"criterion": "entropy"}, rfc_kws={"max_features": "auto"})

    violating_samples = ptree._get_violating_samples_mask()
    scores, fig = ptree.evaluate_fit(plot_roc=True)    

    leaves = ptree.export_leaves(extract_rules_kws={"clause_joiner": None})
    violating_leaves = [leaf for leaf in leaves if leaf['is_violating']]
    violating_consistencies = [leaf['consistency'] for leaf in leaves if leaf['is_violating']]
    

    transcript = transcribe(violating_leaves, var_encoding=var_encoding)
    
    timestamp = str(int(time.time()))
    fig.savefig(folder + 'figures/positivitree' + timestamp + '.pdf')
    
    return scores, X_df[violating_samples].index, transcript, leaves