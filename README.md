# Readme

The codebase contains techniques for identifying positivity violation. It also contains code for estimating the effect of injectables on discontinuation and characterizing overlap regions.

`data` contains the datasets

- `<ctry_code>_Preprocessed.csv` - preprocessed file - including censored records.
- `<ctry_code>_transcript.txt` - support overlap transcript
- `encoding.csv`  - selected features and their encoding
- `fp_select<timestamp>.csv` - encoded and feature selected subset of the data for each country in the following order (et, ng, sl, br, zm, lb, ug)
- `varencoding-<ctry_code>` - DHS encoding of the features

`figures` contains the plots for all the experiments

- `causaleval<timestamp>.pdf` - evaluation plot ordered by ipw; (et, ng, sl, br, zm, lb, ug), ipw overlap (et, ng, sl, br, zm, lb, ug), ow;(et, ng, sl, br, zm, lb, ug), ow overlap (et, ng, sl, br, zm, lb, ug)
- `effects<timestamp>.pdf` - distribution of ATE in the same order as above
- `outcomes<timestamp>.pdf` - distribution of marginal effect in the same order as above
- `placeboeffects<timestamp>.pdf` - distribution of placebo effect in the same order as above
- `supportaccuracy<timestamp>.pdf` - plot of hyperparameter search in the same country order
- `supportclause<>timestamp.pdf` - plot of number of literals vs number of clauses
- `rulesets.pptx` rulesets figures in editable form

`mdscan` - Multidimensional Subset Scanning technique ([link](https://github.ibm.com/AIScience/mdscan))

`notebooks` - some notebooks for positivity violation experimentation - e.g wanted to see if we could apply subset scanning (1d) to the bottleneck layer of an auto-encoder to identify deviations between treated and control groups. The embedding was also useful for applying IRM.

`overrule` - OverRule: Overlap Estimation using Rule Sets ([link](https://github.com/clinicalml/overlap-code))

`positivitree` - Positivitree: Finding and characterizing positivity violations using decision trees ([link](https://github.ibm.com/MLHLS/PositiviTree))

`OW.py` - implementation of overlap weighting

`comp_causalmodel.py` - causal modelling methods - effect estimation and evaluation

`comp_overrule.py` - overrule methods for support ruleset estimation, getting the optimal parsimonious hyper parameter, and helper for getting index of overlap violations

`comp_overrule_clr.py`  - overrule methods for propensity overlap ruleset estimation - with calibrated logistic regression

`comp_overrule_knn.py`  - overrule methods for propensity overlap ruleset estimation - with knn

`comp_postivitree.py` - positivitree methods for learning rulesets

`comp_preprocessing.py` - methods for encoding the covariates and and obtaining the DHS encoding

`hypsearch.npy` - full results from Overrule hyper-parameter search

`overrule_exps.ipynb` overrule experiments 

`ptree_exps.ipynb` causal modelling and positivitree experiments

`riskratios.txt` ATEs in ratio form - for all the experiments

`utils.py` a number of helper functions to transcribe rulesets, to read or write models or files
