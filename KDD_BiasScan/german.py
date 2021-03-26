from matplotlib_venn import venn2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import json
from tqdm import tqdm

from BiasScan.optim.bisection_bias import bisection_q_mle

from BiasScan.solver.bisection_bias import bisection_q_min, bisection_q_max
from BiasScan.score_bias import compute_qs_bias, score_bias
from BiasScan.MDSS import MDSS

scanner = MDSS(
    optim_q_mle=bisection_q_mle,
    solver_q_min=bisection_q_min,
    solver_q_max=bisection_q_max,
    compute_qs=compute_qs_bias,
    score=score_bias
)

with open('datasets/GERMAN.slices.json') as fp:
    german_json = json.load(fp)
    fp.close()

german_scan = pd.read_csv('datasets/german_scan.csv')
german = pd.read_csv('datasets/GERMAN.csv')

data = []
for i in range(len(german_json['data'])):
    
    for j in range(len(german_json['data'][i]['values'])):
        
        pval = german_json['data'][i]['values'][j]['pvalue']
        rank = german_json['data'][i]['values'][j]['rank']
        rule = german_json['data'][i]['values'][j]['rule']
        
        data.append({'pval': pval, 'rule': rule, 'rank': rank})

sorted_data = sorted(data, key=lambda k: k['rank'], reverse=True) 

def get_rule_index(rules):
    setlist = []
    for rule in rules:
        rule = rule.strip() + '.index'
        s = set()
        s.update(list(eval(rule)))
        setlist.append(s)

    intersection = set.intersection(*setlist)
    
    return intersection
    
rules  = sorted_data[0]['rule'].replace('tmp', 'german').strip().replace('german = ', '').split('\t')
freeai_index = get_rule_index(rules)

coordinates = german_scan[german_scan.columns[:-2]]
penalty = 1e-15
direction='negative'
subset = scanner.run_bias_scan(coordinates=coordinates,
        outcomes=german_scan['observed'],
        probs=german_scan['expectation'],
        penalty=penalty,
        num_iters=10,
        num_threads=1,
        direction=direction)


dummy_subset = dict({'index': range(len(freeai_index))})
coordinates = pd.DataFrame(dummy_subset, index=freeai_index)

mdssscore = scanner.score_current_subset(coordinates=coordinates,
            probs=german_scan.iloc[list(freeai_index)]['expectation'],
            outcomes=german_scan.iloc[list(freeai_index)]['observed'],
            penalty=penalty,
            current_subset=dummy_subset,
            direction=direction)

print("Bias Scan HS Subset: \n", subset)
print("FreeAI HR Slice: \n", rules)

mdss_index = set(german_scan[german_scan[subset[0].keys()].isin(subset[0]).all(axis=1)].index)
subset_df = german_scan[german_scan[subset[0].keys()].isin(subset[0]).all(axis=1)]

print("Length of Bias Scan HS Subset: \n", len(subset_df))
print("Mean Obs of Bias Scan HS Subset: \n", subset_df['observed'].mean())

print("Length of FreeAI HR Slice: \n", len(freeai_index))
print("Mean Obs of FreeAI HR Slice: \n", german_scan.iloc[list(freeai_index)]['observed'].mean())
print("Bias Score of FreeAI HR Slice: \n", mdssscore)


set1 = set(freeai_index)
set2 = set(mdss_index)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
venn2([set1, set2], ('FreeAI', 'MDSS-BiasScan'), ax=ax)
plt.show()

mdssscores = []
for idx, data in tqdm(enumerate(sorted_data)):
    rules  = data['rule'].replace('tmp', 'german').strip().replace('german = ', '').split('\n\t')
    _freeai_index = get_rule_index(rules)

    dummy_subset = dict({'index': range(len(_freeai_index))})
    _coordinates = pd.DataFrame(dummy_subset, index=_freeai_index)

    _mdssscore = scanner.score_current_subset(coordinates=_coordinates,
                probs=german_scan.iloc[list(_freeai_index)]['expectation'],
                outcomes=german_scan.iloc[list(_freeai_index)]['observed'],
                penalty=penalty,
                current_subset=dummy_subset,
                direction=direction)
    mdssscores.append(_mdssscore)

plt.hist(mdssscores)
plt.axvline(subset[1], color='k', linestyle='dashed', linewidth=1)
plt.axvline(mdssscore, color='r', linestyle='dashed', linewidth=1)


min_ylim, max_ylim = plt.ylim()
plt.text(subset[1] * 1.1, max_ylim * 0.9, ' Bias Scan: {:.2f}'.format(subset[1]))
plt.text(mdssscore * 1.1, max_ylim * 0.8, ' Freeai: {:.2f}'.format(mdssscore))

plt.xlabel('bias score')
plt.ylabel('number of freeai rules')
plt.show()
