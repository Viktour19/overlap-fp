import numpy as np
from scipy import stats
# from sklearn.metrics import log_loss as cross_entropy
from collections import namedtuple


def positivitree_hypgeom(N0, N1, n0, n1):
    HypGeom = namedtuple("HypGeom", ["pmf", "cdf", "sf"])
    pmf = stats.hypergeom.pmf(n1, N0 + N1, N1, n0+n1)
    cdf = stats.hypergeom.cdf(n1, N0 + N1, N1, n0+n1)
    sf = stats.hypergeom.sf(n1, N0 + N1, N1, n0+n1)
    results = HypGeom(pmf, cdf, sf)
    return results


def positivitree_entropy(N0, N1, n0, n1):
    N = N0 + N1
    n = n0 + n1
    H_root = stats.entropy([N0/N, N1/N], base=2)
    H_leaf = stats.entropy([n0/n, n1/n], base=2)
    Entropy = namedtuple("Entropy", ["root", "leaf", "ratio", "diff"])
    results = Entropy(H_root, H_leaf, H_leaf / H_root, H_leaf - H_root)
    return results


def positivitree_prop_test(N0, N1, n0, n1):
    P = N1 / (N0+N1)
    p = n1 / (n0+n1)
    se = np.sqrt(P * (1-P) / (N0+N1))
    z = (p-P) / se
    abs_z = abs(z)
    left_tail = stats.norm.cdf(-abs_z) 
    right_tail = stats.norm.sf(abs_z)
    two_tail = left_tail + right_tail
    PropTestResults = namedtuple("PropTest", ["P", "p", "se", "z", "left_tail", "right_tail", "two_tail"])
    results = PropTestResults(P, p, se, z, left_tail, right_tail, two_tail)
    return results


def positivitree_fisher(N0, N1, n0, n1):
    results = stats.fisher_exact([[n0, n1], [N0, N1]])
    return results
