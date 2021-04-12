from matplotlib_venn import venn2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import json
from tqdm import tqdm

from BiasScan.optim.bisection_bias import bisection_q_mle
from BiasScan.solver.bisection_bias import bisection_q_min, bisection_q_max
from BiasScan.score_bias import compute_qs_bias, score_bias

# from BiasScan.optim.bisection_poisson import bisection_q_mle
# from BiasScan.solver.bisection_poisson import bisection_q_min, bisection_q_max
# from BiasScan.score_poisson import compute_qs_poisson, score_poisson

from BiasScan.MDSS import MDSS

from rich import print
from itertools import combinations

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

german_scan['observed'] = 1 - german_scan['observed']

for col in german_scan.columns:
    if 'bin' in col:
        s_idx = np.argsort([float(x.split('-')[0]) for x in german_scan[col].unique()])
        v_sorted = german_scan[col].unique()[s_idx]
        german_scan[col] = pd.Categorical(german_scan[col].values, categories=v_sorted, ordered=True)

coordinates = german_scan[german_scan.columns[:-2]]
penalty = 0.01
direction='positive'

data = []
for i in range(len(german_json['data'])):
    
    for j in range(len(german_json['data'][i]['values'])):

        pval = german_json['data'][i]['values'][j]['pvalue']
        support = german_json['data'][i]['values'][j]['support']
        num_mispred = german_json['data'][i]['values'][j]['num_mispred']
        rank = german_json['data'][i]['values'][j]['rank']
        rule = german_json['data'][i]['values'][j]['rule']
        
        data.append({'pval': pval, 'rule': rule, 'rank': rank, 'prop': num_mispred / support})

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

comb_list = list(combinations(coordinates.columns, 3))

max_score = -1
subset = None
for comb in tqdm(comb_list[:1]):
    subset_ = scanner.run_bias_scan(coordinates=coordinates[list(comb)],
            outcomes=german_scan['observed'],
            probs=german_scan['expectation'],
            penalty=penalty,
            num_iters=10,
            num_threads=1,
            direction=direction)
    
    if max_score < subset_[1]:
        subset = subset_
        max_score = subset_[1]

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
prop_hr_slice = german_scan.iloc[list(freeai_index)]['observed'].mean()
print("Mean Obs of FreeAI HR Slice: \n", prop_hr_slice)
print("Bias Score of FreeAI HR Slice: \n", mdssscore)


set1 = set(freeai_index)
set2 = set(mdss_index)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
venn2([set1, set2], ('FreeAI', 'MDSS-BiasScan'), ax=ax)
plt.show()

# mdssscores = []
# props = []
# for idx, data in tqdm(enumerate(sorted_data)):
#     rules  = data['rule'].replace('tmp', 'german').strip().replace('german = ', '').split('\n\t')
#     _freeai_index = get_rule_index(rules)

#     dummy_subset = dict({'index': range(len(_freeai_index))})
#     _coordinates = pd.DataFrame(dummy_subset, index=_freeai_index)

#     _mdssscore = scanner.score_current_subset(coordinates=_coordinates,
#                 probs=german_scan.iloc[list(_freeai_index)]['expectation'],
#                 outcomes=german_scan.iloc[list(_freeai_index)]['observed'],
#                 penalty=penalty,
#                 current_subset=dummy_subset,
#                 direction=direction)
#     props.append(data['prop'])
#     mdssscores.append(_mdssscore)


# plt.hist(mdssscores, bins=50)
# plt.axvline(subset[1], color='k', linestyle='dashed', linewidth=1)
# plt.axvline(mdssscore, color='r', linestyle='dashed', linewidth=1)


# min_ylim, max_ylim = plt.ylim()
# plt.text(subset[1] * 1.01, max_ylim * 0.9, ' Bias Scan: {:.2f}'.format(subset[1]))
# plt.text(mdssscore * 1.01, max_ylim * 0.8, ' Freeai: {:.2f}'.format(mdssscore))

# plt.xlabel('bias score')
# plt.ylabel('number of freeai rules')
# plt.show()

# # plt.hist(props)
# # plt.axvline(prop_hr_slice, color='k', linestyle='dashed', linewidth=1)
# # plt.axvline(german_scan['expectation'].mean(), color='r', linestyle='dashed', linewidth=1)


# # min_ylim, max_ylim = plt.ylim()
# # plt.text(german_scan['expectation'].mean() * 1.1, max_ylim * 0.9, ' Global: {:.2f}'.format(german_scan['expectation'].mean()))
# # plt.text(prop_hr_slice * 1.1, max_ylim * 0.8, ' Freeai: {:.2f}'.format(prop_hr_slice))

# # plt.xlabel('proportion of mispred')
# # plt.ylabel('number of freeai rules')
# # plt.show()

# def remove_subset(subset, dataframe):

#     mask = dataframe[subset[0].keys()].isin(subset[0]).all(axis=1)
#     dataframe = dataframe.loc[~mask]
#     return dataframe

# print(subset)
# subset_df = german_scan[german_scan[subset[0].keys()].isin(subset[0]).all(axis=1)]

# print("Length of HS Subset: \n", len(subset_df))
# print("Mean Obs of HS Subset: \n", subset_df['observed'].mean())

# k = 5
# _german_scan = german_scan.copy()
# for i in tqdm(range(k)):

#     _german_scan = remove_subset(subset, _german_scan)
#     _coordinates = _german_scan[_german_scan.columns[:-2]]

#     subset = scanner.run_bias_scan(coordinates=_coordinates,
#             outcomes=_german_scan['observed'],
#             probs=_german_scan['expectation'],
#             penalty=penalty,
#             num_iters=10,
#             num_threads=1,
#             direction=direction)

# print(subset)
# subset_df = _german_scan[_german_scan[subset[0].keys()].isin(subset[0]).all(axis=1)]

# print("Length of HS Subset: \n", len(subset_df))
# print("Mean Obs of HS Subset: \n", subset_df['observed'].mean())

# k = 100
# random_scores = []

# for i in tqdm(range(k)):

#     np.random.seed(i)
#     german_scan['observed'] = pd.Series(np.random.binomial(1, german_scan['expectation'].mean(), len(german_scan)), \
#         index=german_scan.index)
#     coordinates = german_scan[german_scan.columns[:-2]]
    
#     subset = scanner.run_bias_scan(coordinates=coordinates,
#             outcomes=german_scan['observed'],
#             probs=german_scan['expectation'],
#             penalty=penalty,
#             num_iters=50,
#             num_threads=1,
#             direction=direction)

#     random_scores.append(subset[1])

# plt.hist(random_scores)
# plt.xlabel('bias score')
# plt.ylabel('count')
# plt.show()