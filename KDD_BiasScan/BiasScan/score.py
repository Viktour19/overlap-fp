import numpy as np


def q_dscore(observed_sum, probs, q):
    """
    This actually computes q times the slope, which has the same sign as the slope since q is positive.
    score = Y log q - \sum_i log(1-p_i+qp_i)
    dscore/dq = Y/q - \sum_i (p_i/(1-p_i+qp_i))
    q dscore/dq = Y - \sum_i (qp_i/(1-p_i+qp_i))

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param q: current value of q
    :return: q dscore/dq
    """
    return observed_sum - (q * probs / (1 - probs + q * probs)).sum()


def dq_dscore(probs, q):
    return -(probs * (1 - probs) / (1 - probs + q * probs)**2).sum()


def ddscore(observed_sum: float, probs: np.array, q: float):
    """
    Computes second derivative of bias score given q

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param q: current value of q
    :return: second derivative of bias score for the current value of q
    """
    return -observed_sum / (q ** 2) + ((probs ** 2) / (1 - probs + q * probs)).sum()


def dscore(observed_sum: float, probs: np.array, q: float):
    """
    Computes first derivative of bias score given q

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param q: current value of q
    :return: first derivative of bias score for the current value of q
    """
    return observed_sum / q - (probs / (1 - probs + q * probs)).sum()


def score(observed_sum: float, probs: np.array, penalty: float, q: float):
    """
    Computes bias score for given q

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. Should be positive
    :param q: current value of q
    :return: bias score for the current value of q
    """

    assert q > 0, "Warning: calling compute_score_given_q with " \
                  "observed_sum=%.2f, probs of length=%d, penalty=%.2f, q=%.2f" \
                  % (observed_sum, len(probs), penalty, q)

    return observed_sum * np.log(q) - np.log(1 - probs + q * probs).sum() - penalty

