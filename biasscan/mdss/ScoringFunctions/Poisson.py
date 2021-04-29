from mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from mdss.ScoringFunctions import optim

import numpy as np

class Poisson(ScoringFunction):

    def __init__(self, **kwargs):
        
        super(Poisson, self).__init__()
        self.kwargs = kwargs

    def score(self, observed_sum: float, probs: np.array, penalty: float, q: float):
        """
        Computes poisson bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assert q > 0, "Warning: calling compute_score_given_q with " \
                    "observed_sum=%.2f, probs of length=%d, penalty=%.2f, q=%.2f" \
                    % (observed_sum, len(probs), penalty, q)

        return observed_sum * np.log(q) + (probs - q *probs).sum()  - penalty

    def qmle(self, observed_sum: float, probs: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        assert 'direction' in self.kwargs.keys()
        direction = self.kwargs['direction']
        return optim.bisection_q_mle(self, observed_sum, probs, direction=direction)

    def compute_qs(self, observed_sum: float, probs: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param penalty: penalty coefficient
        """

        direction = None
        if 'direction' in self.kwargs:
            direction = self.kwargs['direction']

        q_mle = self.qmle(observed_sum, probs)

        if self.score(observed_sum, probs, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, probs, penalty, q_mle)
            q_max = optim.bisection_q_max(self, observed_sum, probs, penalty, q_mle)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        return exist, q_mle, q_min, q_max


    def q_dscore(self, observed_sum, probs, q):
        """
        This actually computes q times the slope, which has the same sign as the slope since q is positive.
        score = Y log q + \sum_i (p_i - qp_i)
        dscore/dq = Y / q - \sum_i(p_i)
        q dscore/dq = q_dscore = Y - (q * \sum_i(p_i))

        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param q: current value of q
        :return: q dscore/dq
        """
        return observed_sum - (q * probs).sum()