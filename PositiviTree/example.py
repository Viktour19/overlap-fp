from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# from positivitree.positivitree import PositiviTree
from positivitree.tree_vis import visualize_leaves_static, visualize_leaves_interactive
from positivitree import PositiviTree


def plot_data(X, y, violating_samples=None, single_col_legend=True):
    color_treated, color_untreated = "C0", "C1"
    marker_violating = "P"
    if violating_samples is None:
        violating_samples = np.zeros_like(y, dtype=bool)

    bins = np.histogram(np.ravel(X), bins="auto")[1]
    treated = y == 1
    # Get joint-plot graph structure:
    g = sns.jointplot(x=[], y=[], dropna=False)
    # g.ax_joint.legend_.remove()

    # Plot group:
    _plot_group(X[treated], bins, color_treated, g, marker_violating, violating_samples[treated])
    _plot_group(X[~treated], bins, color_untreated, g, marker_violating, violating_samples[~treated])

    legend_handles = [plt.Line2D([], [], color=color_untreated, alpha=0.6,
                                 marker="o", linestyle="None", label="Group 0"),
                      plt.Line2D([], [], color=color_treated, alpha=0.6,
                                 marker="o", linestyle="None", label="Group 1")]
    ncol = 2
    if np.any(violating_samples):
        legend_handles += [plt.Line2D([], [], color="black", alpha=0.7, marker=marker_violating,
                                      linestyle="None", label="Flagged")]
        ncol += 1
    g.ax_joint.legend(handles=legend_handles, loc="lower left", ncol=ncol,
                      handletextpad=-0.2, columnspacing=0.2, borderpad=0.13, borderaxespad=0.1)
    if single_col_legend:
        g.ax_joint.legend(handles=legend_handles, loc="lower left", ncol=1, handletextpad=0)
    return g


def _plot_group(X, bins, color, g, marker_violating="P", violating_samples=None):
    marginal_type = dict(hist=True, kde=False, rug=False)

    # Plot marginal x:
    sns.distplot(X[:, 0], bins, **marginal_type, color=color, ax=g.ax_marg_x)
    # Plot marginal y:
    sns.distplot(X[:, 1], bins, **marginal_type, color=color, vertical=True, ax=g.ax_marg_y)
    # Plot joint:
    fill_color = colors.colorConverter.to_rgba(color, alpha=0.3)
    edge_color = colors.colorConverter.to_rgba(color, alpha=0.8)
    if violating_samples is not None:
        g.ax_joint.scatter(X[violating_samples, 0], X[violating_samples, 1],
                           edgecolor=edge_color, facecolor=fill_color, marker=marker_violating)
        X = X[~violating_samples]
    # sns.regplot(X[:, 0], X[:, 1], fit_reg=False, ax=g.ax_joint,
    #             scatter_kws=dict(edgecolor=edge_color, facecolor=fill_color))
    g.ax_joint.scatter(X[:, 0], X[:, 1], edgecolor=edge_color, facecolor=fill_color)


def plot_prop_dist(prob, group):
    fig, ax = plt.subplots()
    for group_val in np.unique(group):
        group_val = int(group_val)
        show_group_val = int(group_val)
        show_group_val = int((group_val + 1) % 2)   # For diamond
        sns.distplot(prob[group == group_val], ax=ax,
                     label=f"Group {group_val}", color=f"C{show_group_val}",
                     hist=False, kde_kws={'shade': True})
    ax.legend(loc="best")
    ax.set_xlabel("Propensity")
    ax.set_ylabel("Density")
    return ax


def generate_outlier_data():
    X, y = make_blobs(n_samples=[500, 500, 8],
                      centers=np.array([[5, 5],
                                        [5, 5],
                                        [10, 10]]),
                      cluster_std=0.5,
                      shuffle=False)
    y[y == 2] = 1
    return X, y


def generate_truncated_unit_circle_data(n_samples=2500):
    np.random.seed(42)
    # Generate circle:
    theta = 2 * np.pi * np.random.random(n_samples)
    radius = np.sqrt(np.random.random(n_samples))
    x_0 = radius * np.cos(theta)
    x_1 = radius * np.sin(theta)
    # Generate labels:
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1
    # Remove all first quadrant of negatives:
    first_quadrant_mask = (x_0 > 0) & (x_1 > 0)
    untreated_mask = y == 0
    filter_out = first_quadrant_mask & untreated_mask
    x_0, x_1, y = x_0[~filter_out], x_1[~filter_out], y[~filter_out]
    X = np.column_stack((x_0, x_1))
    return X, y


def generate_truncated_diamond_data(n_samples=2500, angle=45):
    np.random.seed(42)
    # Generate diamond
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    X = np.random.uniform(low=-1, high=1, size=(n_samples, 2))
    X = X @ rotation_matrix
    # Generate labels:
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1
    # Remove all first quadrant of negatives:
    first_quadrant_mask = (X[:, 0] > 0) & (X[:, 1] > 0)
    untreated_mask = y == 0
    filter_out = first_quadrant_mask & untreated_mask
    X, y = X[~filter_out], y[~filter_out]
    return X, y


def load_NHEFS_data():
    import pandas as pd
    data = pd.read_csv("../data/nhefs.csv")
    features = [
        'sex', 'race', 'age', 'education',
        'smokeintensity', 'smokeyrs', 'exercise', 'active', 'wt71'
    ]

    # data = data[restriction_cols]
    y = data["qsmk"]
    X = data[features]

    # X = data.loc[:, data.isnull().mean(axis="index") < 0.1]
    # X.drop(columns=["wt82", "wt82_71"], inplace=True)
    # X = X.dropna()
    # y = X.pop("qsmk")
    return X, y


def run_simulated_data(data_type="diamond"):
    if data_type == "diamond":
        X, y = generate_truncated_diamond_data()
    elif data_type == "circle":
        X, y = generate_truncated_unit_circle_data()
    else:
        X, y = generate_outlier_data()

    ptree = PositiviTree(X, y,
                         violation_cutoff=0.1, consistency_cutoff=0.6,
                         n_consistency_tests=200,
                         search_kws=False,
                         dtc_kws={"criterion": "entropy"},
                         rfc_kws={"max_features": "auto"}
                         )
    ptree = PositiviTree(X, y,
                         violation_cutoff=0.1, relative_violations=False,
                         consistency_cutoff=0.6, n_consistency_tests=200,
                         dtc_kws={"criterion": "entropy"},
                         rfc_kws={"max_features": "auto"},
                         search_kws=True
                         )

    # fit_scores = ptree.evaluate_fit()
    # flagged_leaves = ptree._flag_out_leaves()
    violating_samples = ptree._get_violating_samples_mask()
    # counts = ptree._count_violating_samples_in_forest(normalize=True)

    g = plot_data(X, y, violating_samples, single_col_legend=data_type == "diamond")
    # g.ax_joint.set_xlabel("$x_1$")
    # g.ax_joint.set_ylabel("$x_2$")
    # plot_data(X, y)

    # scores = ptree.evaluate_fit(plot_roc=True)
    # plt.figure()
    # plt.hist(counts)
    # [print(k, v) for k, v in dict(zip(X.columns, ptree.rfc.feature_importances_)).items()]
    tree_json = ptree.export_tree()
    # query_14 = ptree._extract_rule_from_node(14)
    violating_queries = ptree.extract_rules_from_violating_leaves()
    # plt.figure(figsize=(8, 3))
    # plt.figure(figsize=(6, 2.25))
    # visualization_data = viz_tree(ptree)
    # visualization_data = visualize_leaves_static(ptree)
    # visualization_data = visualize_leaves_static(ptree, mix_colors=False)
    # visualization_data = visualize_leaves_interactive(ptree, mix_colors=False)
    visualization_data = ptree.visualize(interactive=True, mix_colors=False)
    # visualization_data = visualize_leaves_interactive(ptree, mix_colors=True)
    # from pandas import DataFrame
    # visualization_data = DataFrame(visualization_data).to_csv()
    # plot_prop_dist(ptree.dtc.predict_proba(X)[:, 1], y)
    print("End.")


def run_NHEFS_data():
    np.random.seed(42)
    X, y = load_NHEFS_data()
    y = y.replace({1: 0, 0: 1})

    ptree = PositiviTree(X, y,
                         violation_cutoff=0.1, consistency_cutoff=0.6,
                         n_consistency_tests=200, relative_violations=False,
                         # search_kws={"n_iter": 200},
                         dtc_kws={"criterion": "entropy"},
                         rfc_kws={"max_features": "auto"}
                         )

    flagged_leaves = ptree._flag_out_leaves()
    violating_samples = ptree._get_violating_samples_mask()
    counts = ptree._count_violating_samples_in_forest(normalize=True)
    tree_json = ptree.export_tree()
    scores = ptree.evaluate_fit(plot_roc=True)

    # sns.pairplot(X.astype(float).join(y), hue=y.name)
    # X_reduce = TSNE().fit_transform(X)
    # plot_data(X_reduce, y, violating_samples)

    # plt.figure()
    # viz_tree(ptree)
    # visualize_leaves_static(ptree)
    visualize_leaves_static(ptree, mix_colors=False)
    # visualization_data = visualize_leaves_interactive(ptree, mix_colors=True)
    # visualization_data = visualize_leaves_interactive(ptree, mix_colors=False)

    plt.figure()
    plt.hist(counts)
    # [print(k, v) for k, v in dict(zip(columns, ptree.rfc.feature_importances_)).items()]
    # plot_prop_dist(ptree.dtc.predict_proba(X)[:, 1], y)
    print("End.")


if __name__ == '__main__':
#     run_simulated_data()
    run_NHEFS_data()

