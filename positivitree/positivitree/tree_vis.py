"""
(C) IBM Corp, 2019, All rights reserved
Created on Feb 28, 2019

@author: EHUD KARAVANI
"""
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mpl_colors
    matplotlib_exists = True
except ImportError:
    matplotlib_exists = False
try:
    from bokeh import plotting, colors as bk_colors, palettes
    from bokeh.models import Range1d
    bokeh_exists = True
except ImportError:
    bokeh_exists = False


def visualize_leaves(ptree, interactive=True, **kwargs):
    if interactive:
        return visualize_leaves_interactive(ptree, **kwargs)
    else:
        return visualize_leaves_static(ptree, **kwargs)


def visualize_leaves_interactive(ptree, plot_width=800, plot_height=400, mix_colors=False,
                                 palette=None, output_path="positivitree_viz.html"):
    if not bokeh_exists:
        warnings.warn("Bokeh could not be loaded. Interactive visualization is skipped.", ImportWarning)
        return
    palette = palette or palettes.Category10[10]  # type: list
    # palette = palette[:2][::-1]     # diamond example
    # palette = palette or palettes.Set2[8]  # type: list
    plotting.output_file(output_path)  # #

    # Data:
    leaves = ptree.export_leaves(extract_rules_kws=dict(clause_joiner=" and ", decimals=3))  # list of dicts
    leaves = sorted(leaves, key=lambda x: x["depth"])

    # Group colors:
    assert palette[0].startswith("#") and len(palette[0]) == 7  # Of form: "#ab1234"
    group_colors = dict(zip(leaves[0]["group_count"].keys(), palette))
    group_colors = {group: (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16))
                    for group, hex_color in group_colors.items()}
    group_colors = {group: bk_colors.RGB(*rgb_color)
                    for group, rgb_color in group_colors.items()}

    # add rectangle location and sizes location of rectangles:
    left_ptr = 0
    for leaf in leaves:
        leaf["w"] = sum(leaf["group_count"].values())
        leaf["h"] = 1
        leaf["y"] = leaf["depth"]               # transform so (x, y) are the centers
        leaf["x"] = left_ptr + (leaf["w"] / 2)  # transform so (x, y) are the centers
        leaf["opacity"] = leaf["consistency"] + np.random.uniform(low=0, high=0.10)
#         leaf["opacity"] = leaf["consistency"] 
        
        if mix_colors:
            counts = [c for c in leaf["group_count"].values()]
            color = [(c.to_hsl().h, c.to_hsl().s, c.to_hsl().l) for c in group_colors.values()]
            color = np.average(color, axis=0, weights=counts)
            color = bk_colors.HSL(*color).to_rgb()
        else:
            majority = max(leaf["group_count"].items(), key=lambda x: x[1])[0]
            color = group_colors[majority]
        leaf["color"] = color

        leaf["group_count"] = str(leaf["group_count"])  # convert dict to string so it is displayable in hover-tool

        left_ptr += leaf["w"]
    # Convert to columnar structure:
    leaves = {attr: [leaf[attr] for leaf in leaves] for attr in leaves[0].keys()}  # dict of lists
    source = plotting.ColumnDataSource(data=leaves)

    tooltips = [("{}".format(attr), "@{}".format(attr)) for attr in leaves.keys()
                if attr not in {"y", "x", "w", "h", "color", "opacity"}]

    p = plotting.figure(width=plot_width, height=plot_height, tooltips=tooltips,
                        title="PositiviTree",
                        x_axis_label="Samples (grouped by leaf)", y_axis_label="Leaf depth")

    p.rect(x="x", y="y", width="w", height="h",
           line_alpha=1.0, line_color="color",
           fill_alpha="opacity", fill_color="color",
           source=source)

    p.xgrid.visible = False
    p.ygrid.visible = False
    p.yaxis.minor_tick_line_color = None
    p.yaxis.ticker = np.arange(min(leaves["depth"]), max(leaves["depth"]) + 1)
    p.y_range = Range1d(min(leaves["depth"]) - 1, max(leaves["depth"]) + 1)
    # Legend:
    for group, color in group_colors.items():
        p.rect(x=0, y=0, width=0, height=0,         # Dummy assignment
               fill_color=color, line_color=color, legend=f"Group {group}")
    p.legend.location = 'bottom_right'
    # plotting.show(p)
    plotting.save(p)
    return p


def visualize_leaves_static(ptree, mix_colors=False, ax=None):
    if not matplotlib_exists:
        warnings.warn("Matplotlib could not be loaded. Static visualization is skipped.", ImportWarning)
        return
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    leaves = ptree.export_leaves()  # list of dicts

    leaves = sorted(leaves, key=lambda x: x["depth"])
    left_ptr = 0
    # max_depth = 0
    for leaf in leaves:
        width = sum(leaf["group_count"].values())

        if mix_colors:
            # Weighted average of colors by their counts:
            hsv_colors = [mpl_colors.rgb_to_hsv(mpl_colors.to_rgb(f"C{c}")) for c in leaf["group_count"].keys()]
            # hsv_colors = hsv_colors[::-1]   # diamond example
            counts = [c for c in leaf["group_count"].values()]
            color = np.average(hsv_colors, axis=0, weights=counts)
            color = mpl_colors.hsv_to_rgb(color)
            # # Using divergent color gradient (purple to orange) and "mix colors" from that line:
            # color = leaf["group_count"][1] / sum(leaf["group_count"].values()) * plt.get_cmap("PuOr").N
            # color = plt.get_cmap("PuOr")(color)
        else:
            majority = max(leaf["group_count"].items(), key=lambda x: x[1])[0]
            # majority = min(leaf["group_count"].items(), key=lambda x: x[1])[0]  # Only for simulation example
            color = f"C{majority}"

#         opacity = leaf["consistency"]
        opacity = leaf["consistency"] + np.random.uniform(low=0, high=0.10)   # Boosting colors in NHEFS

        rectangle = plt.Rectangle(xy=(left_ptr, leaf["depth"]-0.5), width=width, height=1,
                                  fill=True, alpha=opacity, color=color)
        ax.add_patch(rectangle)
        left_ptr += width
    ax.set_xlabel("Samples (grouped by leaves)")
    ax.set_ylabel("Leaf depth")
    ax.set_xlim(0, left_ptr)
    ax.set_ylim(leaves[0]["depth"] - 0.9, leaves[-1]["depth"] + 0.9)    # `leaves` are sorted by depth
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.show()
    return ax


def plot_roc_curve(positivitree, X=None, y=None, sample_weight=None):
    if not matplotlib_exists:
        warnings.warn("Matplotlib could not be loaded. ROC-curve plot is skipped.", ImportWarning)
        return
    X = positivitree.X if X is None else X
    y = positivitree.y if y is None else y
    fig, ax = plt.subplots()
    for estimator_name, estimator in zip(["PositiviTree", "Forest"],
                                         [positivitree.dtc, positivitree.rfc]):
        y_score = estimator.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score, sample_weight=sample_weight)
        auc = roc_auc_score(y, y_score, sample_weight=sample_weight)
        ax.plot(fpr, tpr, label=r"{} (AUC={:.2f}%)".format(estimator_name, auc * 100))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance", zorder=1)
    ax.legend(loc="lower right")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PositiviTree ROC Curve")
    return fig


