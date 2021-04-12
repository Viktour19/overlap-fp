# KDD_BiasScan

This code combines work from Robert's summer internsihp and recent improvements by DB Neil for allowing complexity constraints.

We attempt to quantify how different classifiers respond to bias induced on the dataset that they are trained on.
We consider 4 scenarios of possible bias.

1)  Small training data sets.  We (uniformly) reduce the number of rows in the training data set.
2)  Ommitted/protected variables.  We remove some features from the training set (i.e. gender) but still allow that feature in the scanning data.
3)  Covariate shift.  We choose a random subset (i.e. hispanic females) and then reduce the number of rows _from that subset_ in the training data.  This changes the distribution of features between training and testing.
4)  Transfer learning.  We chooes a random subset (i.e. hispanic females) and then increase the P(Y = 1) for that subset in the training set.  This creates a transfer learning problem between training and test set.


