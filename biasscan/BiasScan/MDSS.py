from typing import Callable
from multiprocessing import cpu_count, Pool
import operator
import logging
import sys
import time

from BiasScan.util.generator import *
from BiasScan.optim.bisection_bias import *

from BiasScan.solver.bisection_bias import *
from BiasScan.solver.bisection_bj import *

from BiasScan.util.contiguous_feature import get_contiguous_set_indices
from pandas.api.types import CategoricalDtype


# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

class ContiguousSubset:
    def __init__(self, observed_sum, probs, penalty, names):
        self.observed_sum = observed_sum
        self.probs  = probs
        self.penalty = penalty
        self.feature_values = names

class MDSS(object):
    def __init__(self, optim_q_mle: Callable, solver_q_min: Callable, solver_q_max: Callable, compute_qs: Callable, score: Callable, alpha: float= None, missing_label = 'missing'):
        self.optim_q_mle = optim_q_mle
        self.solver_q_min = solver_q_min
        self.solver_q_max = solver_q_max
        self.compute_qs = compute_qs
        self.score = score
        self.alpha = alpha
        self.missing_label = missing_label
    

    def get_aggregates(self, coordinates: pd.DataFrame, outcomes: pd.Series, probs: pd.Series,
                       current_subset: dict, column_name: str, penalty: float, direction: str ='positive'):
        """
        Conditioned on the current subsets of values for all other attributes,
        compute the summed outcome (observed_sum = \sum_i y_i) and all probabilities p_i
        for each value of the current attribute.
        Also use additive linear-time subset scanning to compute the set of distinct thresholds
        for which different subsets of attribute values have positive scores. Note that the number
        of such thresholds will be linear rather than exponential in the arity of the attribute.

        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param current_subset: current subset to compute aggregates
        :param column_name: attribute name to scan over
        :param penalty: penalty coefficient
        :param direction: direction of bias positive/negative
        :return: dictionary of aggregates, sorted thresholds (roots), observed sum of the subset, array of observed
        probabilities
        """

        # compute the subset of records matching the current subgroup along all other dimensions
        # temp_df includes the covariates x_i, outcome y_i, and predicted probability p_i for each matching record
        if current_subset:
            to_choose = coordinates[current_subset.keys()].isin(current_subset).all(axis=1)
            temp_df = pd.concat([coordinates.loc[to_choose], outcomes[to_choose], probs[to_choose]], axis=1)
        else:
            temp_df = pd.concat([coordinates, outcomes, probs], axis=1)

        # these wil be used to keep track of the aggregate values and the distinct thresholds to be considered
        aggregates = {}
        thresholds = set()

        # consider each distinct value of the given attribute (column_name)
        for name, group in temp_df.groupby(column_name):
            # compute the sum of outcomes \sum_i y_i
            observed_sum = group.iloc[:, -2].sum()

            # all probabilities p_i
            probs = group.iloc[:, -1].values

            # compute q_min and q_max for the attribute value
            exist, q_mle, q_min, q_max = self.compute_qs(self.optim_q_mle, self.solver_q_min, self.solver_q_max,
             observed_sum, probs, penalty, direction=direction, alpha=self.alpha)

            # Add to aggregates, and add q_min and q_max to thresholds.
            # Note that thresholds is a set so duplicates will be removed automatically.
            if exist:
                aggregates[name] = {
                    'q_mle': q_mle,
                    'q_min': q_min,
                    'q_max': q_max,
                    'observed_sum': observed_sum,
                    'probs': probs
                }
                thresholds.update([q_min, q_max])

        # We also keep track of the summed outcomes \sum_i y_i and the probabilities p_i for the case where _
        # all_ values of that attribute are considered (regardless of whether they contribute positively to score).
        # This is necessary because of the way we compute the penalty term: including all attribute values, equivalent
        # to ignoring the attribute, has the lowest penalty (of 0) and thus we need to score that subset as well.
        all_observed_sum = temp_df.iloc[:, -2].sum()
        all_probs = temp_df.iloc[:, -1].values

        return [aggregates, sorted(thresholds), all_observed_sum, all_probs]

    def choose_aggregates(self, aggregates: dict, thresholds: list, penalty: float, all_observed_sum: float,
                          all_probs: list, direction: str='positive'):
        """
        Having previously computed the aggregates and the distinct q thresholds
        to consider in the get_aggregates function,we are now ready to choose the best
        subset of attribute values for the given attribute.
        For each range defined by these thresholds, we will choose all of the positive contributions,
        compute the MLE value of q, and the corresponding score.
        We then pick the best q and score over all of the ranges considered.

        :param aggregates: dictionary of aggregates. For each feature value, it has q_mle, q_min, q_max, observed_sum,
        and the probabilities
        :param thresholds: sorted thresholds (roots)
        :param penalty: penalty coefficient
        :param all_observed_sum: sum of observed binary outcomes for all i
        :param all_probs: data series containing all the probabilities/expected outcomes
        :param direction: direction of bias positive/negative
        :return:
        """
        # initialize
        best_score = 0
        best_names = []

        # for each threshold
        for i in range(len(thresholds) - 1):
            threshold = (thresholds[i] + thresholds[i + 1]) / 2
            observed_sum = 0.0
            probs = []
            names = []

            # keep only the aggregates which have a positive contribution to the score in that q range
            # we must keep track of the sum of outcome values as well as all predicted probabilities
            for key, value in aggregates.items():
                if (value['q_min'] < threshold) & (value['q_max'] > threshold):
                    names.append(key)
                    observed_sum += value['observed_sum']
                    probs = probs + value['probs'].tolist()

            if len(probs) == 0:
                continue

            # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
            probs = np.asarray(probs)
            current_q_mle = self.optim_q_mle(observed_sum, probs, direction=direction, alpha=self.alpha)

            # Compute the score for the given subset at the MLE value of q.
            # Notice that each included value gets a penalty, so the total penalty
            # is multiplied by the number of included values.
            current_score = self.score(observed_sum=observed_sum, probs=probs, penalty=penalty * len(names), q=current_q_mle, alpha=self.alpha)

            # keep track of the best score, best q, and best subset of attribute values found so far
            if current_score > best_score:
                best_score = current_score
                best_names = names

        # Now we also have to consider the case of including all attribute values,
        # including those that never make positive contributions to the score.
        # Note that the penalty term is 0 in this case.  (We are neglecting penalties
        # from all other attributes, just considering the current attribute.)

        # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
        current_q_mle = self.optim_q_mle(all_observed_sum, all_probs, direction=direction, alpha=self.alpha)

        # Compute the score for the given subset at the MLE value of q.
        # Again, the penalty (for that attribute) is 0 when all attribute values are included.
        
        current_score = self.score(observed_sum=all_observed_sum, probs=all_probs, penalty=0, q=current_q_mle, alpha=self.alpha)

        # Keep track of the best score, best q, and best subset of attribute values found.
        # Note that if the best subset contains all values of the given attribute,
        # we return an empty list for best_names.
        if current_score > best_score:
            best_score = current_score
            best_names = []

        return [best_names, best_score]

    def choose_connected_aggregates(self, aggregates: dict, thresholds: list, penalty: float, all_observed_sum: float, all_probs: list, direction: str='positive', attribute_to_scan = None):

        missing_label = self.missing_label
        no_missing_label = '~' + missing_label

        contiguous_set_indices = get_contiguous_set_indices(attribute_to_scan)
        all_feature_values = list(attribute_to_scan.cat.categories)

        best_names = []
        best_score = -1e10
        
        
        for contiguous_subset in contiguous_set_indices:
            observed_sum = 0.0
            probs = []
            curr_penalty = penalty
            #make a dictionary of subsets with and without missing.
            
            contiguous_subset_dict = {}

            contiguous_subset_dict[missing_label] = ContiguousSubset(observed_sum, probs, penalty, '')
            contiguous_subset_dict[no_missing_label] = ContiguousSubset(observed_sum, probs, penalty, '')
            
            for feature_value_index in contiguous_subset:
                feature_value = all_feature_values[feature_value_index]
                
                if feature_value in aggregates.keys():
                    
                    observed_sum += aggregates[feature_value]['observed_sum']
                    probs = probs + aggregates[feature_value]['probs'].tolist()
            
            contiguous_subset_dict[no_missing_label].observed_sum = observed_sum
            contiguous_subset_dict[no_missing_label].probs = probs
            contiguous_subset_dict[no_missing_label].feature_values = [all_feature_values[i] for i in contiguous_subset]

            # check if we will include Missing Value or not.
            
            if missing_label in aggregates.keys():
                #have score of two subsets to be checked

                curr_penalty= penalty * 2
                observed_sum += aggregates[missing_label]['observed_sum']
                probs = probs + aggregates[missing_label]['probs'].tolist()
                
                contiguous_subset_dict[missing_label].observed_sum = observed_sum
                contiguous_subset_dict[missing_label].probs = probs
                contiguous_subset_dict[missing_label].feature_values = [all_feature_values[i] for i in contiguous_subset]

                
                contiguous_subset_dict[missing_label].feature_values.append(missing_label)
                contiguous_subset_dict[missing_label].penalty = curr_penalty
                
            else:
                curr_penalty = penalty * 2
                contiguous_subset_dict[missing_label].feature_values = [all_feature_values[i] for i in contiguous_subset]
                contiguous_subset_dict[missing_label].feature_values.append(missing_label)
                contiguous_subset_dict[missing_label].observed_sum = observed_sum
                contiguous_subset_dict[missing_label].probs = probs
                contiguous_subset_dict[missing_label].penalty = curr_penalty
                
                            
            for key in contiguous_subset_dict.keys():
                #calculate score for both missing and not missing case
                #need to convert list of probs into an array.

                observed_sum = contiguous_subset_dict[key].observed_sum
                probs = np.array(contiguous_subset_dict[key].probs)
                current_q_mle = self.optim_q_mle(observed_sum, probs, direction=direction, alpha=self.alpha)
                
                current_score = self.score(observed_sum=observed_sum, probs=probs, penalty=contiguous_subset_dict[key].penalty, q=current_q_mle, alpha=self.alpha)

                if current_score > best_score:
                    best_names = contiguous_subset_dict[key].feature_values
                    best_score = current_score
 
        #cover just the missing case:
        observed_sum = 0.0
        probs = []
        curr_penalty = penalty

        if missing_label in aggregates.keys():
            observed_sum += aggregates[missing_label]['observed_sum']
            probs = probs + aggregates[missing_label]['probs'].tolist()

        probs = np.array(probs)

        current_q_mle = self.optim_q_mle(observed_sum, probs, direction=direction, alpha=self.alpha)
        current_score = self.score(observed_sum=observed_sum, probs=probs, penalty=curr_penalty, q=current_q_mle, alpha=self.alpha)
        
        if current_score >= best_score:
            best_names = [missing_label]
            best_score = current_score

    
        #cover the all case:
        current_q_mle = self.optim_q_mle(all_observed_sum, all_probs, direction=direction, alpha=self.alpha)
        current_score = self.score(observed_sum=all_observed_sum, probs=all_probs, penalty=0, q=current_q_mle, alpha=self.alpha)

        if current_score >= best_score:
            best_names = []
            best_score = current_score
  
        return [best_names, best_score]
     

    # score_current subset
    def score_current_subset(self, coordinates: pd.DataFrame, probs: pd.Series, outcomes: pd.Series,
                             penalty: float, current_subset: dict, direction='positive'):
        """
        Just scores the subset without performing ALTSS.
        We still need to determine the MLE value of q.

        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param penalty: penalty coefficient
        :param current_subset: current subset to be scored
        :param direction: direction of bias positive/negative
        :return: penalized score of subset
        """

        # compute the subset of records matching the current subgroup along all dimensions
        # temp_df includes the covariates x_i, outcome y_i, and predicted probability p_i for each matching record
        if current_subset:
            to_choose = coordinates[current_subset.keys()].isin(current_subset).all(axis=1)
            temp_df = pd.concat([coordinates.loc[to_choose], outcomes[to_choose], probs[to_choose]], axis=1)
        else:
            temp_df = pd.concat([coordinates, outcomes, probs], axis=1)

        # we must keep track of the sum of outcome values as well as all predicted probabilities
        observed_sum = temp_df.iloc[:, -2].sum()
        probs = temp_df.iloc[:, -1].values

        # compute the MLE value of q, making sure to only consider the desired direction (positive or negative)
        current_q_mle = self.optim_q_mle(observed_sum, probs, direction=direction, alpha=self.alpha)

        # total_penalty = penalty * sum of list lengths in current_subset
        total_penalty = 0

        # need to change to cater to fact that contiguous value count penalty once
        for key, values in current_subset.items():
            if coordinates[key].dtype.str == CategoricalDtype.str:
                # no need to cover the ALL case since the feature with all values included will not be key
                if self.missing_label in values and len(values) == 1:
                    total_penalty += 1
                elif self.missing_label in values:
                    total_penalty += 2
                else:
                    total_penalty += 1
            else:
                total_penalty += len(values)

        total_penalty *= penalty

        # Compute and return the penalized score    
        penalized_score = self.score(observed_sum=observed_sum, probs=probs, penalty=total_penalty, q=current_q_mle, alpha=self.alpha)
        return penalized_score

    def bias_scan(self, coordinates: pd.DataFrame, outcomes: pd.Series, probs: pd.Series,
                  penalty: float, num_iters: int, direction: str = 'positive', verbose: bool = False,
                  seed: int = 0, thread_id: int = 0):
        """
        :param coordinates: data frame containing having as columns the covariates/features
        :param probs: data series containing the probabilities/expected outcomes
        :param outcomes: data series containing the outcomes/observed outcomes
        :param penalty: penalty coefficient
        :param num_iters: number of iteration
        :param direction: direction of bias positive/negative
        :param verbose: logging flag
        :param seed: numpy seed. Default equals 0
        :param thread_id: id of the worker thread
        :return: [best subset, best score]
        """
        np.random.seed(seed)

        # initialize
        best_subset = {}
        best_score = -1e10

        for i in range(num_iters):
            # flags indicates that the method has optimized over subsets for a given attribute.
            # The iteration ends when it cannot further increase score by optimizing over
            # subsets of any attribute, i.e., when all flags are 1.
            flags = np.empty(len(coordinates.columns))
            flags.fill(0)

            # Starting subset. Note that we start with all values for the first iteration
            # and random values for succeeding iterations.
            current_subset = get_entire_subset() if (i == 0 and thread_id == 0) \
                else get_random_subset(coordinates, np.random.rand(1).item(), 10)

            # score the entire population
            current_score = self.score_current_subset(
                coordinates=coordinates,
                probs=probs,
                outcomes=outcomes,
                penalty=penalty,
                current_subset=current_subset,
                direction=direction
            )

            while flags.sum() < len(coordinates.columns):

                # choose random attribute that we haven't scanned yet
                attribute_number_to_scan = np.random.choice(len(coordinates.columns))
                while flags[attribute_number_to_scan]:
                    attribute_number_to_scan = np.random.choice(len(coordinates.columns))
                attribute_to_scan = coordinates.columns.values[attribute_number_to_scan]

                # clear current subset of attribute values for that subset
                if attribute_to_scan in current_subset:
                    del current_subset[attribute_to_scan]

                # call get_aggregates and choose_aggregates to find best subset of attribute values
                aggregates, thresholds, all_observed_sum, all_probs = self.get_aggregates(
                    coordinates=coordinates,
                    outcomes=outcomes,
                    probs=probs,
                    current_subset=current_subset,
                    column_name=attribute_to_scan,
                    penalty=penalty,
                    direction=direction
                )

                if coordinates[attribute_to_scan].dtype.str == CategoricalDtype.str:                    
                    temp_names, temp_score = self.choose_connected_aggregates(
                        aggregates=aggregates,
                        thresholds=thresholds,
                        penalty=penalty,
                        all_observed_sum=all_observed_sum,
                        all_probs=all_probs,
                        attribute_to_scan = coordinates[attribute_to_scan]              
                    )
                else:
                    temp_names, temp_score = self.choose_aggregates(
                        aggregates=aggregates,
                        thresholds=thresholds,
                        penalty=penalty,
                        all_observed_sum=all_observed_sum,
                        all_probs=all_probs,
                        direction=direction
                    )

                temp_subset = current_subset.copy()
                # if temp_names is not empty (or null)
                if temp_names:
                    temp_subset[attribute_to_scan] = temp_names

                # Note that this call to score_current_subset ensures that
                # we are penalizing complexity for all attribute values.
                # The value of temp_score computed by choose_aggregates
                # above includes only the penalty for the current attribute.
                temp_score = self.score_current_subset(
                    coordinates=coordinates,
                    probs=probs,
                    outcomes=outcomes,
                    penalty=penalty,
                    current_subset=temp_subset,
                    direction=direction
                )

                # reset flags to 0 if we have improved score
                if temp_score > current_score + 1E-6:
                    flags.fill(0)

                # TODO: confirm with Skyler: sanity check to make sure score has not decreased
                # assert temp_score >= current_score - 1E-6, \
                #     "WARNING SCORE HAS DECREASED from %.3f to %.3f" % (current_score, temp_score)

                flags[attribute_number_to_scan] = 1
                current_subset = temp_subset
                current_score = temp_score

            # print out results for current iteration
            if verbose:
                print("Subset found on iteration", i + 1, "of", num_iters, "with score", current_score, ":")
                print(current_subset)

            # update best_score and best_subset if necessary
            if current_score > best_score:
                best_subset = current_subset.copy()
                best_score = current_score

                if verbose:
                    print("Best score is now", best_score)

            elif verbose:
                    print("Current score of", current_score, "does not beat best score of", best_score)

        return best_subset, best_score

    def run_bias_scan(self, coordinates: pd.DataFrame, outcomes: pd.Series, probs: pd.Series,
                      penalty: float, num_iters: int, direction: str = 'positive',
                      num_threads: int = cpu_count(), verbose: bool = False):

        if num_threads > 1:
            # define thread pool
            pool = Pool(processes=num_threads)

            # define list of results. Each result is a tuple consisting of (score, subgroup)
            results = []
            seeds = np.random.randint(0, 9999, num_threads)

            thread_iters = num_iters // num_threads
            reminder = num_iters % num_threads

            # send jobs to thread pool
            for i in range(num_threads):
                iters = thread_iters + max(reminder, 0)
                reminder = reminder - 1

                results.append(pool.apply_async(
                    self.bias_scan,
                    (coordinates, outcomes, probs, penalty, iters, direction, False, seeds[i], i)
                ))

            # close thread pool & wait for all jobs to be done
            pool.close()
            pool.join()

            # collect the results
            results = [res.get() for res in results]

            # get the best score and sub-population
            best_subset, best_score = max(results, key=operator.itemgetter(1))

        else:
            # single thread
            best_subset, best_score = self.bias_scan(
                coordinates=coordinates,
                outcomes=outcomes,
                probs=probs,
                penalty=penalty,
                num_iters=num_iters,
                direction=direction,
                verbose=verbose
            )

        return best_subset, best_score


if __name__ == "__main__":
    # prepare data
    Data1 = pd.read_csv("datasets/german_scan.csv")

    for col in Data1.columns:
        if 'bin' in col:
            s_idx = np.argsort([float(x.split('-')[0]) for x in Data1[col].unique()])
            v_sorted = Data1[col].unique()[s_idx]
            Data1[col] = pd.Categorical(Data1[col].values, categories=v_sorted, ordered=True)

    Data2_outcomes = Data1['observed']
    Data2_probs = Data1['expectation']

    # Data2_treatments = Data1.copy()
    # del Data2_treatments['ReoffendedWithinTwoYears']

    # probability_mapping = Data1.groupby('COMPASPredictedDecileScore').mean()['ReoffendedWithinTwoYears'].to_dict()
    # Data2_probs = Data1['COMPASPredictedDecileScore'].map(probability_mapping)
    # Data2_probs.name = "prob"

    # prepare bias scan
    bias_scan_penalty = 1e-17
    bias_scan_num_iters = 10
    
    from BiasScan.solver.bisection_bias import bisection_q_min, bisection_q_max
    from BiasScan.score_bias import compute_qs_bias, score_bias

    scanner = MDSS(
        optim_q_mle=bisection_q_mle,
        solver_q_min=bisection_q_min,
        solver_q_max=bisection_q_max,
        compute_qs=compute_qs_bias,
        score=score_bias
    )

    start = time.time()

    [best_subset, best_score] = scanner.run_bias_scan(
        coordinates=Data1[Data1.columns[:-2]],
        outcomes=Data2_outcomes,
        probs=Data2_probs,
        penalty=bias_scan_penalty,
        num_iters=bias_scan_num_iters,
        direction='negative',
        num_threads=1, # you can set it to 1 to see the difference
        verbose=False
    )

    print("Ellapsed: %.3f\n" % (time.time() - start))
    print(best_score)
    print(best_subset)


