# PositiviTree
A discriminative approach for finding and characterizing violation 
of positivity using decision trees.


## background
Violation of positivity between groups make them non-comparable
and thus invalidates any causal question answered by comparing them.
  
Soft violations can cause the estimator to be highly variable, 
and they too can be found by adjusting the sensitivity of the model.

#### Finding violation using discrimination
To find subspaces where positivity is violated in the high-dimensional
covariate domain, 
we formulate the problem as a classification one.  
If the classifier is able to discriminate between the two groups, 
it means there is some subspace enriched with one group and not the other.

#### Characterizing the violated subspaces
By using a decision tree classifier, 
we can go even further and characterize the violated subspace.  
We do so by traversing over the decision path from root to leaf
(a leaf of interest, able to discriminate well between the groups)
and converting the rules in each node to a query.

#### Significant testing
###### Using a forest
By constructing a random forest, we are able to check if a violation
found (i.e., a discrimination of the groups) is consistent across
many bootstrapped trees or was it random "flank" caught due to the
decision tree overfitting the data-set at hand.

###### Using statistical tests
In addition we can place a probability for each leaf using
hypergeometric distribution.  
