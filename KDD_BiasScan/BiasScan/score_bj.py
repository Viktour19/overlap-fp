from typing import Callable
import numpy as np
import logging

def q_mle(observed_sum: float, probs: np.array, **kwargs):
    """
    Computes the q which maximizes score (q_mle).
    for berk jones this is given to be N_a/N
    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param direction: direction not considered
    :return: q MLE
    """

    assert 'alpha' in kwargs.keys(), "Warning: calling bj qmle without alpha"
    alpha = kwargs['alpha']

    direction = None
    if 'direction' in kwargs:
        direction = kwargs['direction']
    q = observed_sum/len(probs)

    if ((direction == 'positive') & (q < alpha)) | ((direction == 'negative') & (q > alpha)):
        return alpha
    return q


def dscore(observed_sum: float, probs: np.array, q: float):
    """
    Computes first derivative of bias score given q

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param q: current value of q
    :return: first derivative of bias score for the current value of q
    """
    return ((q * len(probs)) - observed_sum)/((q - 1) * q)

def score_bj(observed_sum: float, probs: np.array, penalty: float, q: float, **kwargs):
    """
    Computes berk jones score for given q

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :param penalty: penalty term. Should be positive
    :param q: current value of q
    :param alpha: alpha treshold
    :return: berk jones score for the current value of q
    """
    assert 'alpha' in kwargs.keys(), "Warning: calling bj score without alpha"
    alpha = kwargs['alpha']

    if q < alpha:
        # print("the mle is less than alpha so setting q to alpha.")
        q = alpha

    assert q > 0, "Warning: calling compute_score_given_q with " \
                  "observed_sum=%.2f, probs of length=%d, penalty=%.2f, q=%.2f, alpha=%.3f" \
                  % (observed_sum, len(probs), penalty, q, alpha)
    if q == 1:
        return observed_sum * np.log(q/alpha) - penalty

    return observed_sum * np.log(q/alpha) + (len(probs) - observed_sum) * np.log((1 - q)/(1 - alpha)) - penalty


def compute_qs_bj(optim_q_mle: Callable, solver_q_min: Callable, solver_q_max: Callable, observed_sum: float, probs: np.array, penalty: float, **kwargs):
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
    assert 'alpha' in kwargs.keys(), "Warning: calling compute_qs bj without alpha"
    alpha = kwargs['alpha']
    # compute q_mle
    q_mle = optim_q_mle(observed_sum, probs, direction=None, alpha=alpha)

    # if q_mle is greater the 0, then compute the other two roots
    if score_bj(observed_sum=observed_sum, probs=probs, penalty=penalty, q=q_mle, alpha=alpha) > 0:
        exist = 1
        q_min = solver_q_min(observed_sum, probs, penalty, q_mle, alpha=alpha)
        q_max = solver_q_max(observed_sum, probs, penalty, q_mle, alpha=alpha)
    else:
        # there are no roots
        exist = 0
        q_min = 0
        q_max = 0

    return exist, q_mle, q_min, q_max
    