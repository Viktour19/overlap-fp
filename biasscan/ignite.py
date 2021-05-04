from biasscan.mdss.MDSS import MDSS
from biasscan.mdss.ScoringFunctions.Bernoulli import Bernoulli
from biasscan.mdss.ScoringFunctions.Poisson import Poisson

from matplotlib_venn import venn2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import json
from tqdm import tqdm
from rich import print
from itertools import combinations


with open('datasets/adult.slices.json') as fp:
    dataset_json = json.load(fp)
    fp.close()

datastr = 'dataset'
dataset_scan = pd.read_csv('datasets/adult_scan.csv')
dataset = pd.read_csv('datasets/adult.csv')

# this creates the datastructure for contiguous features for the
# columns with the string 'bin' in their names
contiguous = {}
for col in dataset_scan.columns:
    if 'bin' in col:
        s_idx = np.argsort([float(x.split('-')[0]) for x in dataset_scan[col].unique()])
        v_sorted = dataset_scan[col].unique()[s_idx]
        contiguous[col] = v_sorted.tolist()

print(contiguous)
direction = 'positive'
penalty = 10
num_iters = 10

scoring_function = Bernoulli(direction = direction)
scanner = MDSS(scoring_function)

coords = dataset_scan[dataset_scan.columns[:-2]]
subset = scanner.parallel_scan(coords, dataset_scan['observed'], \
    dataset_scan['expectation'], penalty, num_iters, contiguous=contiguous)

data = []
for i in range(len(dataset_json['data'])):
    
    for j in range(len(dataset_json['data'][i]['values'])):

        pval = dataset_json['data'][i]['values'][j]['pvalue']
        support = dataset_json['data'][i]['values'][j]['support']
        num_mispred = dataset_json['data'][i]['values'][j]['num_mispred']
        rank = dataset_json['data'][i]['values'][j]['rank']
        rule = dataset_json['data'][i]['values'][j]['rule']
        
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

rules  = sorted_data[0]['rule'].replace('tmp', datastr).strip().replace(datastr + ' = ', '').split('\t')
freeai_index = get_rule_index(rules)


dummy_subset = dict({'index': range(len(freeai_index))})
coords = pd.DataFrame(dummy_subset, index=freeai_index)

mdssscore = scanner.score_current_subset(coords, 
            dataset_scan.iloc[list(freeai_index)]['observed'],
            dataset_scan.iloc[list(freeai_index)]['expectation'],
            {}, penalty)

print("Bias Scan HS Subset: \n", subset)
print("FreeAI HR Slice: \n", rules)

mdss_index = set(dataset_scan[dataset_scan[subset[0].keys()].isin(subset[0]).all(axis=1)].index)
subset_df = dataset_scan[dataset_scan[subset[0].keys()].isin(subset[0]).all(axis=1)]

print("Length of Bias Scan HS Subset: \n", len(subset_df))
print("Mean Obs of Bias Scan HS Subset: \n", subset_df['observed'].mean())

print("Length of FreeAI HR Slice: \n", len(freeai_index))
prop_hr_slice = dataset_scan.iloc[list(freeai_index)]['observed'].mean()
print("Mean Obs of FreeAI HR Slice: \n", prop_hr_slice)
print("Bias Score of FreeAI HR Slice: \n", mdssscore)

set1 = set(freeai_index)
set2 = set(mdss_index)

fig, ax = plt.subplots()
venn2([set1, set2], ('FreeAI', 'MDSS-BiasScan'), ax=ax)
fig.savefig('figures/venn.pdf')