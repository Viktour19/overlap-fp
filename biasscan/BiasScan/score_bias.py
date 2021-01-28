import numpy as np
from typing import Callable

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


def score_bias(observed_sum: float, probs: np.array, penalty: float, q: float, **kwargs):
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


def compute_qs_bias(optim_q_mle: Callable, solver_q_min: Callable, solver_q_max: Callable, observed_sum: float, probs: np.array, penalty: float, **kwargs):
    """
    Computes q_mle, q_min, q_max
    :param optim_q_mle mle method
    :param solver_q_min method to find root on lhs of qmle
    :param solver_q_max method to find root on rhs of qmle
    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. Should be positive
    :return: flag that indicates if qmin and qmax exist, qmle, qmin, qmax
    """
    # compute q_mle
    
    direction = None
    if 'direction' in kwargs:
        direction = kwargs['direction']
        
    q_mle = optim_q_mle(observed_sum, probs, direction=direction)

    # if q_mle is greater the 0, then compute the other two roots
    if score_bias(observed_sum=observed_sum, probs=probs, penalty=penalty, q=q_mle) > 0:
        exist = 1
        q_min = solver_q_min(observed_sum, probs, penalty, q_mle)
        q_max = solver_q_max(observed_sum, probs, penalty, q_mle)
    else:
        # there are no roots
        exist = 0
        q_min = 0
        q_max = 0
    # only consider the desired direction, positive or negative
    if exist:
        if direction == 'positive':
            if q_max < 1:
                exist = 0
            elif q_min < 1:
                q_min = 1
        else:
            if q_min > 1:
                exist = 0
            elif q_max > 1:
                q_max = 1

    return exist, q_mle, q_min, q_max
