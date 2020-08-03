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
        q = alpha

    assert q > 0, "Warning: calling compute_score_given_q with " \
                  "observed_sum=%.2f, probs of length=%d, penalty=%.2f, q=%.2f, alpha=%.3f" \
                  % (observed_sum, len(probs), penalty, q, alpha)

    return observed_sum * np.log(q/alpha) + (len(probs) - observed_sum) * np.log((1 - q)/(1 - alpha)) - penalty

