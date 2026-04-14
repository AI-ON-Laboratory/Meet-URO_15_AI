"""Visualize cumulative SHAP values."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors # ADDED
import numpy as np
import pandas as pd

from shap.utils import hclust_ordering
from shap.utils._legacy import LogitLink, convert_to_link
from shap.plots import colors
from shap.plots._labels import labels
from matplotlib.colors import LinearSegmentedColormap # ADDED

colors = ["darkgreen", "yellowgreen", "darkorange"]
cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

# Function to determine color based on feature value --- ADDED
def get_color(value, min_value, max_value):
    colors = ["darkgreen", "yellowgreen", "darkorange"]
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    cmap = LinearSegmentedColormap.from_list("my_colormap", colors)  # Green to orange colormap
    return cmap(norm(value))


def __change_shap_base_value(base_value, new_base_value, shap_values) -> np.ndarray:
    """Shift SHAP base value to a new value. This function assumes that `base_value` and `new_base_value` are scalars
    and that `shap_values` is a two or three dimensional array.
    """
    # matrix of shap_values
    if shap_values.ndim == 2:
        return shap_values + (base_value - new_base_value) / shap_values.shape[1]

    # cube of shap_interaction_values
    main_effects = shap_values.shape[1]
    all_effects = main_effects * (main_effects + 1) // 2
    temp = (base_value - new_base_value) / all_effects / 2  # divided by 2 because interaction effects are halved
    shap_values = shap_values + temp
    # Add the other half to the main effects on the diagonal
    idx = np.diag_indices_from(shap_values[0])
    shap_values[:, idx[0], idx[1]] += temp
    return shap_values


def __decision_plot_matplotlib(
    base_value,
    cumsum,
    ascending,
    feature_display_count,
    features, # features_display before
    feature_names, # feature_names_display before
    n_top, # ADDED
    n_bottom, # ADDED
    highlight,
    plot_color,
    axis_color,
    y_demarc_color,
    xlim,
    alpha,
    color_bar,
    auto_size_plot,
    title,
    show,
    legend_labels,
    legend_location,
):
    """Matplotlib rendering for decision_plot()"""

    # update feature_display_count and feature_names depending on n_top and n_bottom --- ADDED
    if n_top is not None or n_bottom is not None:
        feature_display_count += 1 # One more row for the aggregated features
        feature_names.insert(n_bottom, "Sum of other features") # Label for the aggregated features after the n_bottom features

    # image size
    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(8, feature_display_count * row_height + 1.5)
    
    # Store the min and max value of each feature to display --- ADDED
    min_values = np.min(features, axis=0)
    max_values = np.max(features, axis=0)

    # draw vertical line indicating center
    pl.axvline(x=base_value, color="#999999", zorder=-1)

    # draw horizontal dashed lines for each feature contribution (for the aggregated features, too)
    for i in range(1, feature_display_count):
        pl.axhline(y=i, color=y_demarc_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # initialize highlighting
    linestyle = np.array("-", dtype=object)
    linestyle = np.repeat(linestyle, cumsum.shape[0])
    linewidth = np.repeat(1, cumsum.shape[0])
    if highlight is not None:
        linestyle[highlight] = "-."
        linewidth[highlight] = 2

    # plot each observation's cumulative SHAP values --- MODIFIED
    ax = pl.gca()
    fig = ax.get_figure()
    ax.set_xlim(xlim)
    lines = []

    for i in range(cumsum.shape[0]): # for each instance/patient
        if n_bottom is None and n_top is None:
            for j in range(1, cumsum.shape[1]): # for each feature included in the features to display
                feature_value = features[i, j-1]
                color = get_color(feature_value, min_values[j-1], max_values[j-1])
                o = pl.plot(
                    [cumsum[i, j-1], cumsum[i, j]], [j-1, j], color=color, linewidth=linewidth[i], linestyle=linestyle[i]
                )
                lines.append(o[0])
        else:
            for j in range(1, cumsum.shape[1]):
                if j <= n_bottom: # Plot the lines for the individual features to display, colored by their value
                    feature_value = features[i, j-1]
                    color = get_color(feature_value, min_values[j-1], max_values[j-1])
                if j == n_bottom+1: # Plot the gray line for the aggregated features
                    color='gray'
                if j > n_bottom+1: # Plot the lines for the individual features to display, colored by their value
                    feature_value = features[i, j-2]
                    color = get_color(feature_value, min_values[j-2], max_values[j-2])
                o = pl.plot(
                    [cumsum[i, j-1], cumsum[i, j]], [j-1, j], color=color, linewidth=linewidth[i], linestyle=linestyle[i]
                )
                lines.append(o[0])


    # determine font size. if ' *\n' character sequence is found (as in interaction labels), use a smaller
    # font. we don't shrink the font for all interaction plots because if an interaction term is not
    # in the display window there is no need to shrink the font.
    s = next((s for s in feature_names if " *\n" in s), None)
    fontsize = 13 if s is None else 9

    # if there is a single observation and feature values are supplied, print them. # NOT my case
    y_pos = np.arange(0, feature_display_count + 1)
    if (cumsum.shape[0] == 1) and (features is not None):
        renderer = pl.gcf().canvas.get_renderer()  # type: ignore
        inverter = pl.gca().transData.inverted()
        y_pos = y_pos + 0.5
        for i in range(feature_display_count):
            v = features[0, i]
            if isinstance(v, str):
                v = f"({str(v).strip()})"
            else:
                v = "({})".format(f"{v:,.3f}".rstrip("0").rstrip("."))
            t = ax.text(
                np.max(cumsum[0, i : (i + 2)]),
                y_pos[i],
                "  " + v,
                fontsize=fontsize,
                horizontalalignment="left",
                verticalalignment="center_baseline",
                color="#666666",
            )
            bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
            if bb.xmax > xlim[1]:
                t.set_text(v + "  ")
                t.set_x(np.min(cumsum[0, i : (i + 2)]))
                t.set_horizontalalignment("right")
                bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
                if bb.xmin < xlim[0]:
                    t.set_text(v)
                    t.set_x(xlim[0])
                    t.set_horizontalalignment("left")

    # style axes
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labeltop=True)
    pl.yticks(np.arange(feature_display_count) + 0.5, feature_names, fontsize=fontsize)
    ax.tick_params("x", labelsize=11)
    pl.ylim(0, feature_display_count)
    pl.xlabel(labels["MODEL_OUTPUT"], fontsize=13)

    # draw the color bar - must come after axes styling
    if color_bar:
        m = cm.ScalarMappable(cmap=plot_color)
        m.set_array(np.array([0, 1]))

        # place the colorbar --- MODIFIED
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(labels["FEATURE_VALUE"], size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(alpha)
        cb.outline.set_visible(False)  # type: ignore

    if title:
        # TODO decide on style/size
        pl.title(title)

    if ascending:
        pl.gca().invert_yaxis()

    if legend_labels is not None:
        ax.legend(handles=lines, labels=legend_labels, loc=legend_location)

    if show:
        pl.show()


class DecisionPlotResult:
    """The optional return value of decision_plot.

    The class attributes can be used to apply the same scale and feature ordering to other decision plots.
    """

    def __init__(self, base_value, shap_values, feature_names, feature_idx, xlim):
        """Example
        -------
        Plot two decision plots using the same feature order and x-axis.
        >>> range1, range2 = range(20), range(20, 40)
        >>> r = decision_plot(base, shap_values[range1], features[range1], return_objects=True)
        >>> decision_plot(base, shap_values[range2], features[range2], feature_order=r.feature_idx, xlim=r.xlim)

        Parameters
        ----------
        base_value : float
            The base value used in the plot. For multioutput models,
            this will be the mean of the base values. This will inherit `new_base_value` if specified.

        shap_values : numpy.ndarray
            The `shap_values` passed to decision_plot re-ordered based on `feature_order`. If SHAP interaction values
            are passed to decision_plot, `shap_values` is a 2D (matrix) representation of the interactions. See
            `feature_names` to locate the feature positions. If `new_base_value` is specified, the SHAP values are
            relative to the new base value.

        feature_names : list of str
            The feature names used in the plot in the order specified in the decision_plot parameter `feature_order`.

        feature_idx : numpy.ndarray
            The index used to order `shap_values` based on `feature_order`. This attribute can be used to specify
            identical feature ordering in multiple decision plots.

        xlim : tuple[float, float]
            The x-axis limits. This attributed can be used to specify the same x-axis in multiple decision plots.

        """
        self.base_value = base_value
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.feature_idx = feature_idx
        self.xlim = xlim


def decision(
    base_value: float | np.ndarray, # expected_value
    shap_values: np.ndarray, # shap_vs
    features: np.ndarray | pd.Series | pd.DataFrame | list | None = None, # X_train_imp
    feature_names=None, # no
    feature_order="importance",
    feature_display_range=None,
    n_top=None, # ADDED
    n_bottom=None, # ADDED
    reverse=False, # ADDED
    highlight=None,
    link="identity",
    plot_color=None,
    axis_color="#333333",
    y_demarc_color="#333333",
    alpha=None,
    color_bar=True,
    auto_size_plot=True,
    title=None,
    xlim=None,
    show=True,
    return_objects=False,
    ignore_warnings=False,
    new_base_value=None,
    legend_labels=None,
    legend_location="best",
) -> DecisionPlotResult | None:
    """Visualize model decisions using cumulative SHAP values.

    Each plotted line explains a single model prediction. If a single prediction is plotted, feature values will be
    printed in the plot (if supplied). If multiple predictions are plotted together, feature values will not be printed.
    Plotting too many predictions together will make the plot unintelligible.

    Parameters
    ----------
    base_value : float or numpy.ndarray
        This is the reference value that the feature contributions start from. Usually, this is
        ``explainer.expected_value``.

    shap_values : numpy.ndarray
        Matrix of SHAP values (# features) or (# samples x # features) from
        ``explainer.shap_values()``. Or cube of SHAP interaction values (# samples x
        # features x # features) from ``explainer.shap_interaction_values()``.

    features : numpy.array or pandas.Series or pandas.DataFrame or numpy.ndarray or list
        Matrix of feature values (# features) or (# samples x # features). This provides the values of all the
        features and, optionally, the feature names.

    feature_names : list or numpy.ndarray
        List of feature names (# features). If ``None``, names may be derived from the
        ``features`` argument if a Pandas object is provided. Otherwise, numeric feature
        names will be generated.

    feature_order : str or None or list or numpy.ndarray
        Any of "importance" (the default), "hclust" (hierarchical clustering), ``None``,
        or a list/array of indices.

    feature_display_range: slice or range
        The slice or range of features to plot after ordering features by ``feature_order``. A step of 1 or ``None``
        will display the features in ascending order. A step of -1 will display the features in descending order. If
        ``feature_display_range=None``, ``slice(-1, -21, -1)`` is used (i.e. show the last 20 features in descending order).
        If ``shap_values`` contains interaction values, the number of features is automatically expanded to include all
        possible interactions: N(N + 1)/2 where N = ``shap_values.shape[1]``.

    highlight : Any
        Specify which observations to draw in a different line style. All numpy indexing methods are supported. For
        example, list of integer indices, or a bool array.

    link : str
        Use "identity" or "logit" to specify the transformation used for the x-axis. The "logit" link transforms
        log-odds into probabilities.

    plot_color : str or matplotlib.colors.ColorMap
        Color spectrum used to draw the plot lines. If ``str``, a registered matplotlib color name is assumed.

    axis_color : str or int
        Color used to draw plot axes.

    y_demarc_color : str or int
        Color used to draw feature demarcation lines on the y-axis.

    alpha : float
        Alpha blending value in [0, 1] used to draw plot lines.

    color_bar : bool
        Whether to draw the color bar (legend).

    auto_size_plot : bool
        Whether to automatically size the matplotlib plot to fit the number of features
        displayed. If ``False``, specify the plot size using matplotlib before calling
        this function.

    title : str
        Title of the plot.

    xlim: tuple[float, float]
        The extents of the x-axis (e.g. ``(-1.0, 1.0)``). If not specified, the limits
        are determined by the maximum/minimum predictions centered around base_value
        when ``link="identity"``. When ``link="logit"``, the x-axis extents are ``(0,
        1)`` centered at 0.5. ``xlim`` values are not transformed by the ``link``
        function. This argument is provided to simplify producing multiple plots on the
        same scale for comparison.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    return_objects : bool
        Whether to return a :obj:`DecisionPlotResult` object containing various plotting
        features. This can be used to generate multiple decision plots using the same
        feature ordering and scale.

    ignore_warnings : bool
        Plotting many data points or too many features at a time may be slow, or may create very large plots. Set
        this argument to ``True`` to override hard-coded limits that prevent plotting large amounts of data.

    new_base_value : float
        SHAP values are relative to a base value. By default, this base value is the
        expected value of the model's raw predictions. Use ``new_base_value`` to shift
        the base value to an arbitrary value (e.g. the cutoff point for a binary
        classification task).

    legend_labels : list of str
        List of legend labels. If ``None``, legend will not be shown.

    legend_location : str
        Legend location. Any of "best", "upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center".

    Returns
    -------
    DecisionPlotResult or None
        Returns a :obj:`DecisionPlotResult` object if ``return_objects=True``. Returns ``None`` otherwise (the default).

    Examples
    --------
    Plot two decision plots using the same feature order and x-axis.

        >>> range1, range2 = range(20), range(20, 40)
        >>> r = decision_plot(base, shap_values[range1], features[range1], return_objects=True)
        >>> decision_plot(base, shap_values[range2], features[range2], feature_order=r.feature_idx, xlim=r.xlim)

    See more `decision plot examples here <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html>`_.

    """
    # code taken from force_plot. auto unwrap the base_value
    if isinstance(base_value, np.ndarray) and len(base_value) == 1:
        base_value = base_value[0]

    if isinstance(base_value, list) or isinstance(shap_values, list):
        raise TypeError(
            "Looks like multi output. Try base_value[i] and shap_values[i], or use shap.multioutput_decision_plot()."
        )

    # validate shap_values
    if not isinstance(shap_values, np.ndarray):
        raise TypeError("The shap_values arg is the wrong type. Try explainer.shap_values().")

    # calculate the various dimensions involved (observations, features, interactions, display, etc.)
    if shap_values.ndim == 1: # if monodimensional array
        shap_values = shap_values.reshape(1, -1) # reshape to 2D array with one row and as many columns as necessary
    observation_count = shap_values.shape[0] # number of instances/patients/observations
    feature_count = shap_values.shape[1] # number of features (tot)

    # code taken from force_plot. convert features from other types.
    if isinstance(features, pd.DataFrame): # my case
        if feature_names is None:
            feature_names = features.columns.to_list() # feature_names = list of feature names
        features = features.values # converts the DataFrame into a numpy ndarray
    elif isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = features.index.to_list()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and features.ndim == 1 and feature_names is None:
        feature_names = features.tolist()
        features = None

    # the above code converts features to either None or np.ndarray. if features is something else at this point,
    # there's a problem. - NOT my case
    if not isinstance(features, (np.ndarray, type(None))):
        raise TypeError("The features arg uses an unsupported type.")
    if (features is not None) and (features.ndim == 1):
        features = features.reshape(1, -1)

    # validate/generate feature_names. at this point, feature_names does not include interactions. - NOT my case
    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(feature_count)]
    elif len(feature_names) != feature_count:
        raise ValueError("The feature_names arg must include all features represented in shap_values.")
    elif not isinstance(feature_names, (list, np.ndarray)):
        raise TypeError("The feature_names arg requires a list or numpy array.")

    # transform interactions cube to a matrix and generate interaction names. - NOT my case
    if shap_values.ndim == 3:
        # flatten
        triu_count = feature_count * (feature_count - 1) // 2
        idx_diag = np.diag_indices_from(shap_values[0])
        idx_triu = np.triu_indices_from(shap_values[0], 1)
        a: np.ndarray = np.ndarray((observation_count, feature_count + triu_count), shap_values.dtype)
        a[:, :feature_count] = shap_values[:, idx_diag[0], idx_diag[1]]
        a[:, feature_count:] = shap_values[:, idx_triu[0], idx_triu[1]] * 2
        shap_values = a
        # names
        b: list[str | None] = [None] * shap_values.shape[1]
        b[:feature_count] = feature_names
        for i, row, col in zip(range(feature_count, shap_values.shape[1]), idx_triu[0], idx_triu[1]):
            b[i] = f"{feature_names[row]} *\n{feature_names[col]}"
        feature_names = b
        feature_count = shap_values.shape[1]
        features = None  # Can't use feature values for interactions...

    # determine feature order
    if isinstance(feature_order, list):
        feature_idx = np.array(feature_order)
    elif isinstance(feature_order, np.ndarray):
        feature_idx = feature_order
    elif (feature_order is None) or (feature_order.lower() == "none"):
        feature_idx = np.arange(feature_count)
    elif feature_order == "importance": # my case
        feature_idx = np.argsort(np.sum(np.abs(shap_values), axis=0)) # array containing indexes of features sorted in ASCENDING order of importance
    elif feature_order == "hclust": # my case
        feature_idx = np.array(hclust_ordering(shap_values.transpose()))
    else:
        raise ValueError(
            "The feature_order arg requires 'importance', 'hclust', 'none', or an integer list/array "
            "of feature indices."
        )

    if (feature_idx.shape != (feature_count,)) or (not np.issubdtype(feature_idx.dtype, np.integer)):
        raise ValueError(
            "A list or array has been specified for the feature_order arg. The length must match the "
            "feature count and the data type must be integer."
        )

    # validate and convert feature_display_range to a slice. prevents out of range errors later.
    if feature_display_range is None:
        feature_display_range = slice(-1, -21, -1)  # show last 20 features in descending order (from the most important [-1], to the 20th most important [-20])
    elif not isinstance(feature_display_range, (slice, range)):
        raise TypeError("The feature_display_range arg requires a slice or a range.")
    elif feature_display_range.step not in (-1, 1, None):
        raise ValueError("The feature_display_range arg supports a step of 1, -1, or None.")
    elif isinstance(feature_display_range, range):
        # Negative values in a range are not the same as negs in a slice. Consider range(2, -1, -1) == [2, 1, 0],
        # but slice(2, -1, -1) == [] when len(features) > 2. However, range(2, -1, -1) == slice(2, -inf, -1) after
        # clipping.
        c = np.iinfo(np.integer).min
        feature_display_range = slice(
            feature_display_range.start if feature_display_range.start >= 0 else c,  # should never happen, but...
            feature_display_range.stop if feature_display_range.stop >= 0 else c,
            feature_display_range.step,
        )

    # If just one of n_top or n_bottom is None, set that to 0 --- ADDED
    if (n_top is None and n_bottom is not None) or (n_top is not None and n_bottom is None): # exclusive or
        if n_top is None:
            n_top = 0
        else:
            n_bottom = 0

    # apply new_base_value - NOT my case
    if new_base_value is not None:
        shap_values = __change_shap_base_value(base_value, new_base_value, shap_values)
        base_value = new_base_value

    # -----------------------------------------------------------------------------------------------

    # use feature_display_range to determine which features will be plotted. convert feature_display_range to
    # ascending indices and expand by one in the negative direction. why? we are plotting the change in prediction
    # for every feature. this requires that we include the value previous to the first displayed feature
    # (i.e. i_0 - 1 to i_n).
    d = feature_display_range.indices(feature_count) # d=(start, stop, step) - remember that 'stop' is exclusive!
    ascending = True
    if d[2] == -1:  # The step
        ascending = False
        d = (d[1] + 1, d[0] + 1, 1)
    if reverse:
        feature_idx = feature_idx[::-1] # reverse the order of the features
    shap_values = shap_values[:, feature_idx] # order the shap values according to the feature importance order
    if n_top is None and n_bottom is None:
        feature_display_count = d[1] - d[0] # number of features to display
        if d[0] == 0:
            cumsum: np.ndarray = np.ndarray((observation_count, feature_display_count + 1), shap_values.dtype)
            cumsum[:, 0] = base_value
            cumsum[:, 1:] = base_value + np.nancumsum(shap_values[:, 0 : d[1]], axis=1)
        else:
            cumsum = base_value + np.nancumsum(shap_values, axis=1)[:, (d[0] - 1) : d[1]]
    else:
        feature_display_count = n_top + n_bottom
        cumsum: np.ndarray = np.ndarray((observation_count, feature_display_count + 2), shap_values.dtype)
        cumsum[:, 0] = base_value
        cum_sum_tot = base_value + np.nancumsum(shap_values, axis=1)
        cumsum[:, 1:n_bottom+1] = cum_sum_tot[:, 0 : n_bottom]
        cumsum[:, n_bottom+1:] = cum_sum_tot[:, -n_top-1:]
    
    # Select and sort feature names and features according to the range selected above
    feature_names = np.array(feature_names)
    if n_top is None and n_bottom is None:
        feature_names_display = feature_names[feature_idx[d[0] : d[1]]].tolist()
    elif n_top == 0:
        feature_names_display = feature_names[feature_idx[:n_bottom]].tolist()
    else:
        feature_names_display = np.concatenate((feature_names[feature_idx[:n_bottom]], feature_names[feature_idx[-n_top:]])).tolist()
    feature_names = feature_names[feature_idx].tolist()

    if n_top is None and n_bottom is None:
        features_display = None if features is None else features[:, feature_idx[d[0] : d[1]]]
    elif n_top == 0:
        features_display = features[:, feature_idx[:n_bottom]]
    else:
        features_display = np.concatenate((features[:, feature_idx[:n_bottom]], features[:, feature_idx[-n_top:]]), axis=1)

    # -----------------------------------------------------------------------------------------------

    # throw large data errors
    if not ignore_warnings:
        if observation_count > 2000:
            raise RuntimeError(
                f"Plotting {observation_count} observations may be slow. Consider subsampling or set "
                "ignore_warnings=True to ignore this message."
            )
        if feature_display_count > 200:
            raise RuntimeError(
                f"Plotting {feature_display_count} features may create a very large plot. Set "
                "ignore_warnings=True to ignore this "
                "message."
            )
        if feature_count * observation_count > 100000000:
            raise RuntimeError(
                f"Processing SHAP values for {feature_count} features over {observation_count} observations may be slow. Set "
                "ignore_warnings=True to ignore this "
                "message."
            )

    # convert values based on link and update x-axis extents
    create_xlim = xlim is None
    link = convert_to_link(link)
    base_value_saved = base_value
    if isinstance(link, LogitLink): # NOT my case
        base_value = link.finv(base_value)
        cumsum = link.finv(cumsum)
        if create_xlim:
            # Expand [0, 1] limits a little for a visual margin
            xlim = (-0.02, 1.02)
    elif create_xlim:
        xmin: float = min((cumsum.min(), base_value))
        xmax: float = max((cumsum.max(), base_value))
        # create a symmetric axis around base_value
        n, m = (base_value - xmin), (xmax - base_value)
        if n > m:
            xlim = (base_value - n, base_value + m)
        else:
            xlim = (base_value - m, base_value + m)
        # Adjust xlim to include a little visual margin.
        e = (xlim[1] - xlim[0]) * 0.02
        xlim = (xlim[0] - e, xlim[1] + e)

    # Initialize style arguments
    if alpha is None:
        alpha = 1.0

    if plot_color is None:
        plot_color = cmap

    __decision_plot_matplotlib(
        base_value,
        cumsum,
        ascending,
        feature_display_count,
        features_display,
        feature_names_display,
        n_top, # ADDED
        n_bottom, # ADDED
        highlight,
        plot_color,
        axis_color,
        y_demarc_color,
        xlim,
        alpha,
        color_bar,
        auto_size_plot,
        title,
        show,
        legend_labels,
        legend_location,
    )

    if not return_objects:
        return None

    return DecisionPlotResult(base_value_saved, shap_values, feature_names, feature_idx, xlim)


def multioutput_decision(base_values, shap_values, row_index, **kwargs) -> DecisionPlotResult | None:
    """Decision plot for multioutput models.

    Plots all outputs for a single observation. By default, the plotted base value will be the mean of base_values
    unless new_base_value is specified. Supports both SHAP values and SHAP interaction values.

    Parameters
    ----------
    base_values : list of float
        This is the reference value that the feature contributions start from. Use explainer.expected_value.

    shap_values : list of numpy.ndarray
        A multioutput list of SHAP matrices or SHAP cubes from explainer.shap_values() or
        explainer.shap_interaction_values(), respectively.

    row_index : int
        The integer index of the row to plot.

    **kwargs : Any
        Arguments to be passed on to decision_plot().

    Returns
    -------
    DecisionPlotResult or None
        Returns a DecisionPlotResult object if `return_objects=True`. Returns `None` otherwise (the default).

    """
    # todo: adjust to breaking changes made in #3318
    if not (isinstance(base_values, list) and isinstance(shap_values, list)):
        raise ValueError("The base_values and shap_values args expect lists.")

    # convert arguments to arrays for simpler handling
    base_values = np.array(base_values)
    if not ((base_values.ndim == 1) or (np.issubdtype(base_values.dtype, np.number))):
        raise ValueError("The base_values arg should be a list of scalars.")
    shap_values = np.array(shap_values)
    if shap_values.ndim not in [3, 4]:
        raise ValueError("The shap_values arg should be a list of two or three dimensional SHAP arrays.")
    if shap_values.shape[0] != base_values.shape[0]:
        raise ValueError("The base_values output length is different than shap_values.")

    # shift shap base values to mean of base values
    base_values_mean = base_values.mean()
    for i in range(shap_values.shape[0]):
        shap_values[i] = __change_shap_base_value(base_values[i], base_values_mean, shap_values[i])

    # select the feature row corresponding to row_index
    if (kwargs is not None) and ("features" in kwargs):
        features = kwargs["features"]
        if isinstance(features, np.ndarray) and (features.ndim == 2):
            kwargs["features"] = features[[row_index]]
        elif isinstance(features, pd.DataFrame):
            kwargs["features"] = features.iloc[row_index]

    return decision(base_values_mean, shap_values[:, row_index, :], **kwargs)
