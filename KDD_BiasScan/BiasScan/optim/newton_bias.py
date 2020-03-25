from BiasScan.score_bias import *


def newton_q_mle(observed_sum: float, probs: np.array):
    """
    Computes the q which maximizes score (q_mle).
    Computes q for which slope dscore/dq = 0, using the fact that slope is monotonically decreasing.
    q_mle is computed via newton-raphson.
    This works because the score, as a function of q, is concave.
    So the slope is monotonically decreasing, and q_mle is the unique value for which slope = 0.

    :param observed_sum: sum of observed binary outcomes for all i
    :param probs: predicted probabilities p_i for each data element i
    :return: q MLE
    """
    observed_mean = observed_sum / len(probs)

    if np.isclose(observed_mean, 1):
        return 1e6
    elif np.isclose(observed_mean, 0):
        return 1e-6

    expected_mean = probs.mean()
    q = observed_mean * (1 - expected_mean) / ((1 - observed_mean) * expected_mean)
    df = q_dscore(observed_sum, probs, q)

    while np.abs(df) > 1e-6:
        ddf = dq_dscore(probs, q)
        q = q - df / ddf
        df = q_dscore(observed_sum, probs, q)

    return q
