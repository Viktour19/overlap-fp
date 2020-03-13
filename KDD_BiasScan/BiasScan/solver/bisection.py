from BiasScan.score import *


def bisection_q_min(observed_sum, probs, penalty, q_mle):
    """
    Compute q for which score = 0, using the fact that score is monotonically increasing for q > q_mle.
    q_max is computed via binary search.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the LHS of qmle
    """
    q_temp_min = 1e-6
    q_temp_max = q_mle

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if np.sign(score(observed_sum, probs, penalty, q_temp_mid)) > 0:
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2
        else:
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2

    return (q_temp_min + q_temp_max) / 2


def bisection_q_max(observed_sum: float, probs: np.array, penalty: float, q_mle: float):
    """
    Compute q for which score = 0, using the fact that score is monotonically decreasing for q > q_mle.
    q_max is computed via binary search.
    This works because the score, as a function of q, is concave.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. should be positive
    :param q_mle: q maximum likelihood
    :return: the root on the RHS of qmle
    """
    q_temp_min = q_mle
    q_temp_max = 1e6

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if np.sign(score(observed_sum, probs, penalty, q_temp_mid)) > 0:
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2
        else:
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2

    return (q_temp_min + q_temp_max) / 2
