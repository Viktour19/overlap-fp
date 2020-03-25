from BiasScan.score_bias import *
import numpy as np

def newton_q_min(observed_sum, probs, penalty, q_mle):
    """
    Compute q for which score = 0, using the fact that score is monotonically increasing for q > q_mle.
    q_max is computed via newton-raphson.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the LHS of qmle
    """
    if np.isclose(observed_sum, 0):
        return 1e-6

    q = 1e-6
    f = score(observed_sum, probs, penalty, q)

    while np.abs(f) > 1e-6:
        df = dscore(observed_sum, probs, q)
        q = q - f / df
        f = score(observed_sum, probs, penalty, q)

    return q


def newton_q_max(observed_sum: float, probs: np.array, penalty: float, q_mle: float):
    """
    Compute q for which score = 0, using the fact that score is monotonically decreasing for q > q_mle.
    q_max is computed via newton-raphson.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the RHS of qmle
    """
    if np.isclose(observed_sum, len(probs)):
        return 1e6

    q = q_mle + 1
    f = score(observed_sum, probs, penalty, q)

    while np.abs(f) > 1e-6:
        df = dscore(observed_sum, probs, q)
        q = q - f / df
        f = score(observed_sum, probs, penalty, q)

    return q
