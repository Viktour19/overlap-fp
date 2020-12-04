from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics as sk_metrics
import numpy as np
from scipy import stats

from .tree_vis import plot_roc_curve, visualize_leaves


DEFAULT_IMPURITY_CRITERION = "entropy"
DEFAULT_MIN_SAMPLE_LEAF = 0.001
DEFAULT_SEARCH_PARAMS = {"n_iter": 50,
                         "scoring": ["accuracy", "roc_auc", "average_precision"],
                         "cv": 5,
                         "refit": "roc_auc",
                         "return_train_score": True,
                         "error_score": np.nan}


# TODO: do requirements file


class PositiviTree:
    def __init__(self, X, y, violation_cutoff=0.05, consistency_cutoff=0.5, n_consistency_tests=100,
                 relative_violations=True, search_kws=True, dtc_kws=None, rfc_kws=None):
        """
        Parameters
        ----------
        X : array
            shape: (n_samples, n_features)
            Covariates.
        y : array
            shape: (n_samples,)
            Binary group assignment for each sample.
        violation_cutoff : float
            What validates as a violation of positivity,
             calculated either as H(root) - H(leaf) >= violation_cutoff
             or as H(leaf) <= violation_cutoff
             Depending on the value of relative_violations
        consistency_cutoff : float
            What constitute as a non-random (structural) violation.
             builds a forests from bootstrap samples and counts the frequency of
             which is sample is considered to violate positivity.
             If a sample is was flagged in `consistency_cutoff` percent of the
             trees or more, then it is considered as consistent violator.
        n_consistency_tests: int
            number of consistency tests to apply.
            Basically number of trees to construct in a random forest.
        relative_violations : bool
            Whether to calculate violations relative to root:
             H(root) - H(leaf) >= violation_cutoff
            Or to calculate absolute violations:
             H(leaf) <= violation_cutoff
        search_kws : dict or bool
            Parameters for scikit-learn's RandomizedSearchCV or GridSearchCV.
            Used to find the best generalizable DecisionTreeClassifier.
            If `False`: hyperparameter search is not applied.
            If `True`: hyperparameter search is done with default parameters.
            If `dict`: the parameters for the SearchCV. See the corresponding APIs for details.
            See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        dtc_kws : dict or None
            parameters for scikit-learn's DecisionTreeClassifier.
             Unless provided, max_depth = n_features * 2.
            See: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        rfc_kws : dict or None
            parameters for scikit-learn's RandomForestClassifier or ExtraTreesClassifier
            By design, the forest is built using the same parameters as the main tree.
             If parameters from dtc_kws and rfc_kws contradicts,
             the parameters of rfc_kws prevails.
            Use if you know what your doing.
            See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        self.X = X
        self.y = y
        self.violation_cutoff = violation_cutoff
        self.consistency_cutoff = consistency_cutoff
        self.n_consistency_tests = n_consistency_tests
        self.relative_violations = relative_violations

        self.dtc = self._init_and_fit_dtc(dtc_kws, search_kws)
        self.rfc = self._init_and_fit_rfc(rfc_kws, dtc_kws)

    def _init_and_fit_dtc(self, dtc_kws, search_kws):
        dtc_kws = {} if dtc_kws is None else dtc_kws

        dtc_kws["criterion"] = dtc_kws.get("criterion", DEFAULT_IMPURITY_CRITERION)
        # dtc_kws["min_samples_leaf"] = dtc_kws.get("min_samples_leaf", DEFAULT_MIN_SAMPLE_LEAF)
        dtc_kws["max_depth"] = dtc_kws.get("max_depth", self.X.shape[1] * 2)

        dtc = DecisionTreeClassifier(**dtc_kws)
        dtc.fit(self.X, self.y)

        if search_kws:
            search_kws = search_kws if isinstance(search_kws, dict) else {}     # else search_kws == True
            default_search_params = DEFAULT_SEARCH_PARAMS.copy()
            default_search_params.update(search_kws)
            # Default params:
            param_distributions = {"max_depth": stats.randint(low=int(np.sqrt(self.X.shape[1])),
                                                              high=int(self.X.shape[1]**2)),
                                   "min_samples_split": stats.uniform(loc=0.001, scale=0.1),
                                   "min_samples_leaf": stats.uniform(loc=1/self.X.shape[0], scale=0.1),
                                   # TODO: update scipy -> low & high
                                   # "max_features": [None, "sqrt"]
                                   }
            if "param_distributions" not in DEFAULT_SEARCH_PARAMS:
                default_search_params["param_distributions"] = param_distributions

            self.dtc_search_ = RandomizedSearchCV(dtc, **default_search_params)
            self.dtc_search_.fit(self.X, self.y)
            dtc = self.dtc_search_.best_estimator_
        return dtc

    def _init_and_fit_rfc(self, rfc_kws, dtc_kws):
        rfc_kws = {} if rfc_kws is None else rfc_kws
        dtc_kws = self.dtc.get_params().copy()          # Each tree in forest will ideally be similar to the main DTC

        # Drop DecisionTreeClassifier param that is not relevant for RandomForestClassifier:
#         dtc_kws.pop('presort')
        if dtc_kws.pop('splitter') == "best":
            RandomForest = RandomForestClassifier
        else:   # splitter == "random"
            RandomForest = ExtraTreesClassifier
        # TODO: the default max_features in DTC (all) is different than in RFC (all).
        #       Should we prioritize the same tree-parameters? or the same defaults?
        #       Namely, if a parameter is not provided to DTC and RFC,
        #           do we for the RFC the default of DTC or the default or RFC?

        # Overrun the decision tree params with the ones provided by the user (if any):
        dtc_kws.update(rfc_kws)
        dtc_kws["n_estimators"] = self.n_consistency_tests

        rfc = RandomForest(**dtc_kws)
        rfc.fit(self.X, self.y)
        return rfc

    def visualize(self, interactive=True, **kwargs):
        """See exact kwargs in tree_vis module, depending on whether interactive is used or not"""
        return visualize_leaves(self, interactive, **kwargs)

    def export_leaves(self, dtc=None, consistency_aggregator=np.mean, extract_rules_kws=None):
        # TODO: maybe rewrite independently from export_tree?
        def is_leaf(node):
            return all([l is None for l in node["children"]])

        def extract_leaves(node):
            if is_leaf(node):
                node = {k: v for k, v in node.items()
                        if k not in {"children", "covariate", "threshold"}}     # Remove redundant features
                leaves.append(node)
            else:
                extract_leaves(node["children"][0])
                extract_leaves(node["children"][1])

        tree_dict = self.export_tree(dtc, consistency_aggregator, extract_rules_kws)
        leaves = []
        extract_leaves(tree_dict)
        return leaves

    def export_tree(self, dtc=None, consistency_aggregator=np.mean, extract_rules_kws=None):
        """

        Parameters
        ----------
        dtc : DecisionTreeClassifier, optional
            Any decision-tree-classifier.
            If not provided, the main classifier tree is used.
        consistency_aggregator : callable | None
            Any function taking a vector (ideally reduces it to a scalar, but not enforced).
            Used to aggregate consistency values of all samples belonging to a some node.
            If None: array of consistencies of the provided samples is passed. Be careful with large datasets.
        extract_rules_kws : dict, optional
            Keyword arguments to pass for `_extract_rule_from_node`. See there.

        Returns
        -------
            dict: Nested tree structure with data on each node:
                    node_id, covariate, threshold, depth, group_count, impurity,
                    probability, is_violating, query, consistency, children.
                  "children" key holds a list with [left_child, right_child].
        """
        extract_rules_kws = extract_rules_kws or {}
        dtc = self.dtc if dtc is None else dtc
        sk_tree = dtc.tree_
        feature_names = self.__get_features_name()
        consistency = self._count_violating_samples_in_forest(normalize=True)
        violating_leaves = self._flag_out_leaves(dtc)
        leaves_depth = self._get_nodes_depths(dtc)

        def build_tree(node_id):
            if node_id == -1:
                return None
            node = dict()
            node["node_id"] = node_id
            node["covariate"] = feature_names[sk_tree.feature[node_id]]
            node["threshold"] = sk_tree.threshold[node_id]
            node["depth"] = leaves_depth[node_id]
            node["group_count"] = dict(zip(range(sk_tree.n_classes[0]), sk_tree.value[node_id, 0, :]))
            node["impurity"] = sk_tree.impurity[node_id]
            node["probability"] = self._node_probability(node_id)
            node["is_violating"] = node_id in violating_leaves
            node["query"] = self._extract_rule_from_node(node_id, dtc=dtc, **extract_rules_kws)
            node_consistency = consistency[self._get_samples_id_from_node(node_id, dtc)]
            if consistency_aggregator is not None:
                node_consistency = consistency_aggregator(node_consistency)
            node["consistency"] = node_consistency
            node["children"] = [build_tree(sk_tree.children_left[node_id]),
                                build_tree(sk_tree.children_right[node_id])]
            return node

        tree = build_tree(node_id=0)
        return tree

    def export_sample_level(self, dtc=None):
        """

        Returns
        -------
            (np.ndarray, tuple[str])
            data : np.ndarray
                numpy array shape (n_samples, len(col_names))
            col_names: tuple
                The names corresponding to the columns of the data
                ["id", "group", "consistency", "leaf_id",
                 "leaf_depth", "leaf_impurity", "leaf_probability"]
        """
        dtc = self.dtc if dtc is None else dtc

        # Sample-level-data:
        sample_ids = getattr(self.X, "index").values if hasattr(self.X, "index") else np.arange(self.X.shape[0])
        consistency = self._count_violating_samples_in_forest(normalize=True)
        leaf_assignment = dtc.apply(self.X)

        # Leaf-level-data:
        is_leaf = (dtc.tree_.children_right == -1) & (dtc.tree_.children_left == -1)
        leaf_ids = np.where(is_leaf)[0]
        impurity = dtc.tree_.impurity[leaf_ids]
        probability = np.array([self._node_probability(leaf_id) for leaf_id in leaf_ids])
        depth = self._get_nodes_depths(dtc)[leaf_ids]
        # # convert leaf-level to sample level (join by leaf id)
        leaf_id_to_sample_id_map = dict(zip(leaf_ids, range(leaf_ids.size)))
        leaf_id_to_sample_id_map = np.array(list(map(leaf_id_to_sample_id_map.get, leaf_assignment)))
        impurity = impurity[leaf_id_to_sample_id_map]
        probability = probability[leaf_id_to_sample_id_map]
        depth = depth[leaf_id_to_sample_id_map]

        data = np.column_stack((sample_ids, self.y, consistency,
                                leaf_assignment, depth, impurity, probability))
        col_names = tuple(["id", "group", "consistency", "leaf_id",
                           "leaf_depth", "leaf_impurity", "leaf_probability"])
        return data, col_names

    def _count_violating_samples_in_forest(self, normalize=False, violation_cutoff=None):
        if violation_cutoff is None:
            violation_cutoff = self.violation_cutoff
        impurity_matrix = self._get_forest_impurity()
        violations = impurity_matrix < violation_cutoff
        aggregator = np.mean if normalize else np.sum
        violations = aggregator(violations, axis=1)
        return violations

    def _get_violating_samples_mask(self, dtc=None):
        dtc = self.dtc if dtc is None else dtc

        violating_leaves = self._flag_out_leaves(dtc)
        samples_leaf_assignment = dtc.apply(self.X)
        # Violating samples are those mapped into violating leaves:
        violating_samples_mask = np.isin(samples_leaf_assignment, violating_leaves)
        return violating_samples_mask

    def _flag_out_leaves(self, dtc=None):
        dtc = self.dtc if dtc is None else dtc
        nodes_impurity = dtc.tree_.impurity
        root_impurity = nodes_impurity[0]

        if self.relative_violations:
            are_violating_entropy = root_impurity - nodes_impurity >= self.violation_cutoff
        else:
            are_violating_entropy = nodes_impurity < self.violation_cutoff
        are_leaves = dtc.tree_.children_left == dtc.tree_.children_right
        violating_leaves = are_leaves & are_violating_entropy
        violating_leaves = np.where(violating_leaves)[0]
        return violating_leaves

    def _get_nodes_depths(self, dtc=None):
        dtc = self.dtc if dtc is None else dtc
        nodes_depth = np.zeros(dtc.tree_.node_count)
        stack = [(0, 0)]  # (node_id, node's depth)
        while len(stack):
            node_id, node_depth = stack.pop()
            nodes_depth[node_id] = node_depth
            if dtc.tree_.children_left[node_id] != -1:
                stack.append((dtc.tree_.children_left[node_id], node_depth + 1))
            if dtc.tree_.children_right[node_id] != -1:
                stack.append((dtc.tree_.children_right[node_id], node_depth + 1))
        return nodes_depth

    def _node_probability(self, node_id):
        root_id = 0
        N0 = self.dtc.tree_.value[root_id, 0, 0]    # class 0 counts entire population
        N1 = self.dtc.tree_.value[root_id, 0, 1]    # class 1 counts entire population
        n0 = self.dtc.tree_.value[node_id, 0, 0]    # class 0 counts node's sub-population
        n1 = self.dtc.tree_.value[node_id, 0, 1]    # class 1 counts node's sub-population
        probability = stats.hypergeom.pmf(k=n1, M=N0+N1, n=N1, N=n0+n1)
        return probability

    def _extract_rule_from_node(self, node_id, prune=True, clause_joiner=" AND ", decimals=None, dtc=None):
        """
        Parameters
        ----------
        node_id (int): The ID of the node to extract the rule from root to it.
        prune (bool): If True, gives non-redundant representation of the rule of a node.
                       For example if a path describes the rule ((x>0) and (x>1)), then
                       with prune=True only simplified (x>1) is returned,
                       and with False the entire redundant ((x>0) and (x>1)) is returned.
        clause_joiner (str): format of the operator joining the different rules.
                             Commonly can be 'and', ' AND ' or ' & '.
                             If None - a list of un-joined clauses is returned.
        decimals (int): if clause_joiner is string, then how many decimals to format the threshold value.
        dtc (DecisionTreeClassifier):

        Returns
        -------
            str | list[Rule]: if clause_joiner is not None:
                                formatted string of the rule from root to node,
                              else a list of the individual rules is returned.
        """
        def extract_query_from_path(path):
            rules = []
            for parent_id, node_id in zip(path[:-1], path[1:]):
                single_level_rule = build_single_level_rule(node_id, parent_id)
                if prune:
                    # Since we traverse from root to node, we need to discard repeated rules and
                    # replace them with new ones, as the closer to the node the more specific they
                    # are, as closer to the root the more general they are:
                    rules = [rule for rule in rules
                             if (rule.feature_name != single_level_rule.feature_name) or
                                (rule.threshold_sign != single_level_rule.threshold_sign)]
                rules.append(single_level_rule)
            if clause_joiner is not None:
                rules = [rule.format(decimals) for rule in rules]
                rules_ = []
                if len(rules) > 1:
                    for i in range(len(rules)):
                        rule = rules[i]
                        rules_.append("({})".format(rule))
                        
                        if i != 0 and i % 4 == 0:
                            rules_.append("\n")
                            
#                     rules = ["({})".format(rule) for rule in rules]     # wrap each rule with ()
                rules = clause_joiner.join(rules_)
            return rules

        def build_single_level_rule(node_id, parent_id):
            if dtc.tree_.children_left[parent_id] == node_id:
                threshold_sign = "<="
            elif dtc.tree_.children_right[parent_id] == node_id:
                threshold_sign = ">"
            else:
                return ""
            threshold_value = dtc.tree_.threshold[parent_id]
            feature_name = feature_names[dtc.tree_.feature[parent_id]]
            # single_query = "({}{}{:.5f})".format(feature_name, threshold_sign, threshold_value)
            # single_query = "({}{}{})".format(feature_name, threshold_sign, threshold_value)
            single_query = Rule(feature_name, threshold_sign, threshold_value)
            return single_query

        def get_path_of_nodes_to_node(node_id):
            nodes = []
            while node_id is not None:
                nodes.append(node_id)
                node_id = get_node_parent(node_id)
            assert nodes[-1] == 0   # made it to root
            return nodes[::-1]

        def get_node_parent(node_id):
            if node_id == 0:   # root
                parent_id = None
            elif node_id in dtc.tree_.children_right:
                parent_id = np.where(dtc.tree_.children_right == node_id)[0][0]
            else:
                parent_id = np.where(dtc.tree_.children_left == node_id)[0][0]
            return parent_id

        dtc = self.dtc if dtc is None else dtc
        feature_names = self.__get_features_name()
        path_to_node = get_path_of_nodes_to_node(node_id)
        query = extract_query_from_path(path_to_node)
        return query

    def _get_samples_id_from_node(self, node_id, dtc=None):
        dtc = self.dtc if dtc is None else dtc
        # CSR format has efficient row-slicing, but we slice columns (all samples of a given node) so transpose:
        decision_path = dtc.decision_path(self.X).transpose()
        samples_assigned_to_node = decision_path[node_id]
        samples_assigned_to_node = np.squeeze(samples_assigned_to_node.toarray())
        samples_assigned_to_node = np.where(samples_assigned_to_node)[0]
        return samples_assigned_to_node

    def __get_features_name(self):
        if hasattr(self.X, "columns"):  # X is a pandas DataFrame
            feature_names = self.X.columns
        else:
            feature_names = ["X[:, {}]".format(i) for i in range(self.X.shape[1])]
        return feature_names

    def extract_rules_from_violating_leaves(self, prune=True, clause_joiner=" AND ",
                                            decimals=None, dtc=None):
        # violating_leaves = self._find_consistent_violating_leaves()
        violating_leaves = self._flag_out_leaves(dtc)
        rules = {}
        for leaf_id in violating_leaves:
            query = self._extract_rule_from_node(leaf_id, prune=prune,
                                                 clause_joiner=clause_joiner,
                                                 decimals=decimals, dtc=dtc)
            rules[leaf_id] = query
        return rules

    def extract_sample_ids_from_violating_leaves(self, dtc=None):
        dtc = self.dtc if dtc is None else dtc

        flagged_out_leaves = self._flag_out_leaves()
        leaf_assignment = dtc.apply(self.X)

        violating_samples = {}
        for flagged_out_leaf in flagged_out_leaves:
            leaf_violating_samples = leaf_assignment == flagged_out_leaf
            leaf_violating_samples = np.where(leaf_violating_samples)[0]
            violating_samples[flagged_out_leaf] = leaf_violating_samples
        return violating_samples

    def _find_consistent_violating_samples(self):
        violating_samples_counts = self._count_violating_samples_in_forest(normalize=True)
        significant_samples_mask = violating_samples_counts > self.consistency_cutoff
        # stats.geom(np.mean(counts / n_estimators)).sf(counts) < 0.01 / n_estimators
        significant_samples_idx = np.where(significant_samples_mask)[0]
        return significant_samples_idx

    def _find_consistent_violating_leaves(self):
        # Finds the leaves containing at least 1 consistently violating sample.
        # TODO: is it how we wanna roll? Alternative: take intersection of the above AND violating leaves.
        #       another alternative is to decide on average-consistency values.
        violating_samples = self._find_consistent_violating_samples()
        leaf_assignment = self.dtc.apply(self.X)
        violating_leaves = leaf_assignment[violating_samples]
        violating_leaves = np.unique(violating_leaves)
        return violating_leaves

    def _get_forest_impurity(self):
        impurity_matrix = np.zeros((self.X.shape[0], self.rfc.n_estimators))
        for i, dtc in enumerate(self.rfc.estimators_):
            leaf_assignment = dtc.apply(self.X)  # get leaf assignment in tree for each individual
            impurity_vector = dtc.tree_.impurity[leaf_assignment]  # get the corresponding impurity value of each sample
            impurity_matrix[:, i] = impurity_vector
        return impurity_matrix

    def evaluate_fit(self, X=None, y=None, sample_weight=None, plot_roc=False,
                     metric_names=("roc_auc", "accuracy", "average_precision")):
        """metric names as provided by scikit learn.
        If plot_roc == True then matplotlib figure is returned in addition to the scores.
        """
        X = self.X if X is None else X
        y = self.y if y is None else y
        scores = {}
        for estimator_name, estimator in zip(["DTC", "RFC"], [self.dtc, self.rfc]):
            scores[estimator_name] = {}
            for metric_name in metric_names:
                scorer = sk_metrics.get_scorer(metric_name)
                score = scorer(estimator, X, y, sample_weight=sample_weight)
                scores[estimator_name][metric_name] = score
        if plot_roc:
            fig = plot_roc_curve(self, X, y, sample_weight)
            return scores, fig
        return scores


class Rule:
    def __init__(self, feature_name, threshold_sign, threshold_value):
        self.feature_name = feature_name
        self.threshold_sign = threshold_sign
        self.threshold_value = threshold_value

    def __str__(self):
        s = "({}{}{})".format(self.feature_name, self.threshold_sign, self.threshold_value)
        return s

    def format(self, decimals=3):
        thresh = "{}".format(self.threshold_value)
        if "." in thresh:   # float
            n_decimals = thresh.rsplit(".", 1)[1]
            if decimals is not None and len(n_decimals) > decimals:
                thresh = "{:.{prec}}".format(self.threshold_value, prec=decimals)
        s = "{name}{sign}{thresh}".format(name=self.feature_name, sign=self.threshold_sign,
                                          thresh=thresh)
        return s

    def __format__(self, format_spec):
        return self.format(format_spec)

