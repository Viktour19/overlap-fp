from BiasScan.score_bias import *
import numpy as np


def bisection_q_mle(observed_sum: float, probs: np.array):
    """
    Computes the q which maximizes score (q_mle).
    Computes q for which slope dscore/dq = 0, using the fact that slope is monotonically decreasing.
    q_mle is computed via bisection.
    This works because the score, as a function of q, is concave.
    So the slope is monotonically decreasing, and q_mle is the unique value for which slope = 0.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :return: q MLE
    """
    q_temp_min = 1e-6
    q_temp_max = 1e6

    # print("LEN_PROBS:", len(probs))

    while np.abs(q_temp_max - q_temp_min) > 1e-6:
        q_temp_mid = (q_temp_min + q_temp_max) / 2

        if np.sign(q_dscore(observed_sum, probs, q_temp_mid)) > 0:
            q_temp_min = q_temp_min + (q_temp_max - q_temp_min) / 2
        else:
            q_temp_max = q_temp_max - (q_temp_max - q_temp_min) / 2

    return (q_temp_min + q_temp_max) / 2

