"""Plotting functions for visualizing distributions."""
from numbers import Number
from functools import partial
import math
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from ._base import VectorPlotter
from ._statistics import ECDF, Histogram, KDE
from ._stats.counting import Hist
from .axisgrid import FacetGrid, _facet_docs
from .utils import remove_na, _get_transform_functions, _kde_support, _check_argument, _assign_default_kwargs, _default_color
from .palettes import color_palette
from .external import husl
from .external.kde import gaussian_kde
from ._docstrings import DocstringComponents, _core_docs
__all__ = ['displot', 'histplot', 'kdeplot', 'ecdfplot', 'rugplot', 'distplot']
_dist_params = dict(multiple='\nmultiple : {{"layer", "stack", "fill"}}\n    Method for drawing multiple elements when semantic mapping creates subsets.\n    Only relevant with univariate data.\n    ', log_scale='\nlog_scale : bool or number, or pair of bools or numbers\n    Set axis scale(s) to log. A single value sets the data axis for any numeric\n    axes in the plot. A pair of values sets each axis independently.\n    Numeric values are interpreted as the desired base (default 10).\n    When `None` or `False`, seaborn defers to the existing Axes scale.\n    ', legend='\nlegend : bool\n    If False, suppress the legend for semantic variables.\n    ', cbar='\ncbar : bool\n    If True, add a colorbar to annotate the color mapping in a bivariate plot.\n    Note: Does not currently support plots with a ``hue`` variable well.\n    ', cbar_ax='\ncbar_ax : :class:`matplotlib.axes.Axes`\n    Pre-existing axes for the colorbar.\n    ', cbar_kws='\ncbar_kws : dict\n    Additional parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.\n    ')
_param_docs = DocstringComponents.from_nested_components(core=_core_docs['params'], facets=DocstringComponents(_facet_docs), dist=DocstringComponents(_dist_params), kde=DocstringComponents.from_function_params(KDE.__init__), hist=DocstringComponents.from_function_params(Histogram.__init__), ecdf=DocstringComponents.from_function_params(ECDF.__init__))

class _DistributionPlotter(VectorPlotter):
    wide_structure = {'x': '@values', 'hue': '@columns'}
    flat_structure = {'x': '@values'}

    def __init__(self, data=None, variables={}):
        super().__init__(data=data, variables=variables)

    @property
    def univariate(self):
        """Return True if only x or y are used."""
        return len(self.variables) == 1 and (self.variables.get("x") is not None or self.variables.get("y") is not None)

    @property
    def data_variable(self):
        """Return the variable with data for univariate plots."""
        if self.univariate:
            return "x" if self.variables.get("x") is not None else "y"
        return None

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        return self.variables.get("x") is not None or self.variables.get("y") is not None

    def _add_legend(self, ax_obj, artist, fill, element, multiple, alpha, artist_kws, legend_kws):
        """Add artists that reflect semantic mappings and put then in a legend."""
        handles = []
        labels = []
        for level in self.var_levels.get(self._hue_var, []):
            if fill:
                if element == "bars":
                    handle = plt.Rectangle((0, 0), 0, 0, **artist_kws)
                elif element == "step":
                    handle = plt.Line2D([], [], **artist_kws)
                else:
                    handle = plt.Polygon([(0, 0)], **artist_kws)
            else:
                handle = plt.Line2D([], [], **artist_kws)
            handles.append(handle)
            labels.append(level)

        if handles:
            legend = ax_obj.legend(handles, labels, title=self._hue_var, **legend_kws)
            ax_obj.add_artist(legend)

    def _artist_kws(self, kws, fill, element, multiple, color, alpha):
        """Handle differences between artists in filled/unfilled plots."""
        kws = kws.copy()
        if fill:
            kws.setdefault("facecolor", color)
            if element == "bars":
                kws.setdefault("edgecolor", "none")
            elif element == "step":
                kws.setdefault("edgecolor", color)
            else:
                kws.setdefault("edgecolor", kws["facecolor"])
        else:
            kws.setdefault("color", color)

        if multiple in ["stack", "fill"]:
            kws.setdefault("linewidth", 0)
        elif element == "bars":
            kws.setdefault("edgecolor", color)

        if alpha is not None:
            kws["alpha"] = alpha

        return kws

    def _quantile_to_level(self, data, quantile):
        """Return data levels corresponding to quantile cuts of mass."""
        if not len(data) or quantile is None:
            return []
        percentiles = np.linspace(0, 100, int(quantile) + 1)
        return np.percentile(data, percentiles[1:-1])

    def _cmap_from_color(self, color):
        """Return a sequential colormap given a color seed."""
        from matplotlib.colors import LinearSegmentedColormap
        rgb = mpl.colors.colorConverter.to_rgb(color)
        light_rgb = [1 - (1 - c) * .25 for c in rgb]
        colors = [light_rgb, rgb]
        return LinearSegmentedColormap.from_list("", colors)

    def _default_discrete(self):
        """Find default values for discrete hist estimation based on variable type."""
        discrete = {}
        for var in ["x", "y"]:
            if var in self.variables:
                discrete[var] = variable_type(self.plot_data[var], boolean_type="categorical") != "numeric"
        return discrete

    def _resolve_multiple(self, curves, multiple):
        """Modify the density data structure to handle multiple densities."""
        if multiple == "layer":
            return curves
        elif multiple == "stack":
            return self._stack_curves(curves)
        elif multiple == "fill":
            return self._fill_curves(curves)
        else:
            raise ValueError(f"multiple '{multiple}' is not recognized")

    def _stack_curves(self, curves):
        """Stack a list of density curves."""
        stacked = []
        for curve in curves:
            new_curve = curve.copy()
            if stacked:
                new_curve[:, 1] += stacked[-1][:, 1]
            stacked.append(new_curve)
        return stacked

    def _fill_curves(self, curves):
        """Normalize stacked density curves."""
        stacked = self._stack_curves(curves)
        filled = []
        for i, curve in enumerate(stacked):
            new_curve = curve.copy()
            if i < len(stacked) - 1:
                new_curve[:, 1] /= stacked[-1][:, 1]
            filled.append(new_curve)
        return filled

    def _plot_single_rug(self, sub_data, var, height, ax, kws):
        """Draw a rugplot along one axis of the plot."""
        kws = kws.copy()
        if var == "x":
            kws.update(dict(ymin=0, ymax=height, c=[]))
            ax.vlines(sub_data[var], **kws)
        else:
            kws.update(dict(xmin=0, xmax=height, c=[]))
            ax.hlines(sub_data[var], **kws)
histplot.__doc__ = 'Plot univariate or bivariate histograms to show distributions of datasets.\n\nA histogram is a classic visualization tool that represents the distribution\nof one or more variables by counting the number of observations that fall within\ndiscrete bins.\n\nThis function can normalize the statistic computed within each bin to estimate\nfrequency, density or probability mass, and it can add a smooth curve obtained\nusing a kernel density estimate, similar to :func:`kdeplot`.\n\nMore information is provided in the :ref:`user guide <tutorial_hist>`.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nweights : vector or key in ``data``\n    If provided, weight the contribution of the corresponding data points\n    towards the count in each bin by these factors.\n{params.hist.stat}\n{params.hist.bins}\n{params.hist.binwidth}\n{params.hist.binrange}\ndiscrete : bool\n    If True, default to ``binwidth=1`` and draw the bars so that they are\n    centered on their corresponding data points. This avoids "gaps" that may\n    otherwise appear when using discrete (integer) data.\ncumulative : bool\n    If True, plot the cumulative counts as bins increase.\ncommon_bins : bool\n    If True, use the same bins when semantic variables produce multiple\n    plots. If using a reference rule to determine the bins, it will be computed\n    with the full dataset.\ncommon_norm : bool\n    If True and using a normalized statistic, the normalization will apply over\n    the full dataset. Otherwise, normalize each histogram independently.\nmultiple : {{"layer", "dodge", "stack", "fill"}}\n    Approach to resolving multiple elements when semantic mapping creates subsets.\n    Only relevant with univariate data.\nelement : {{"bars", "step", "poly"}}\n    Visual representation of the histogram statistic.\n    Only relevant with univariate data.\nfill : bool\n    If True, fill in the space under the histogram.\n    Only relevant with univariate data.\nshrink : number\n    Scale the width of each bar relative to the binwidth by this factor.\n    Only relevant with univariate data.\nkde : bool\n    If True, compute a kernel density estimate to smooth the distribution\n    and show on the plot as (one or more) line(s).\n    Only relevant with univariate data.\nkde_kws : dict\n    Parameters that control the KDE computation, as in :func:`kdeplot`.\nline_kws : dict\n    Parameters that control the KDE visualization, passed to\n    :meth:`matplotlib.axes.Axes.plot`.\nthresh : number or None\n    Cells with a statistic less than or equal to this value will be transparent.\n    Only relevant with bivariate data.\npthresh : number or None\n    Like ``thresh``, but a value in [0, 1] such that cells with aggregate counts\n    (or other statistics, when used) up to this proportion of the total will be\n    transparent.\npmax : number or None\n    A value in [0, 1] that sets that saturation point for the colormap at a value\n    such that cells below constitute this proportion of the total count (or\n    other statistic, when used).\n{params.dist.cbar}\n{params.dist.cbar_ax}\n{params.dist.cbar_kws}\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.core.color}\n{params.dist.log_scale}\n{params.dist.legend}\n{params.core.ax}\nkwargs\n    Other keyword arguments are passed to one of the following matplotlib\n    functions:\n\n    - :meth:`matplotlib.axes.Axes.bar` (univariate, element="bars")\n    - :meth:`matplotlib.axes.Axes.fill_between` (univariate, other element, fill=True)\n    - :meth:`matplotlib.axes.Axes.plot` (univariate, other element, fill=False)\n    - :meth:`matplotlib.axes.Axes.pcolormesh` (bivariate)\n\nReturns\n-------\n{returns.ax}\n\nSee Also\n--------\n{seealso.displot}\n{seealso.kdeplot}\n{seealso.rugplot}\n{seealso.ecdfplot}\n{seealso.jointplot}\n\nNotes\n-----\n\nThe choice of bins for computing and plotting a histogram can exert\nsubstantial influence on the insights that one is able to draw from the\nvisualization. If the bins are too large, they may erase important features.\nOn the other hand, bins that are too small may be dominated by random\nvariability, obscuring the shape of the true underlying distribution. The\ndefault bin size is determined using a reference rule that depends on the\nsample size and variance. This works well in many cases, (i.e., with\n"well-behaved" data) but it fails in others. It is always a good to try\ndifferent bin sizes to be sure that you are not missing something important.\nThis function allows you to specify bins in several different ways, such as\nby setting the total number of bins to use, the width of each bin, or the\nspecific locations where the bins should break.\n\nExamples\n--------\n\n.. include:: ../docstrings/histplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
kdeplot.__doc__ = 'Plot univariate or bivariate distributions using kernel density estimation.\n\nA kernel density estimate (KDE) plot is a method for visualizing the\ndistribution of observations in a dataset, analogous to a histogram. KDE\nrepresents the data using a continuous probability density curve in one or\nmore dimensions.\n\nThe approach is explained further in the :ref:`user guide <tutorial_kde>`.\n\nRelative to a histogram, KDE can produce a plot that is less cluttered and\nmore interpretable, especially when drawing multiple distributions. But it\nhas the potential to introduce distortions if the underlying distribution is\nbounded or not smooth. Like a histogram, the quality of the representation\nalso depends on the selection of good smoothing parameters.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nweights : vector or key in ``data``\n    If provided, weight the kernel density estimation using these values.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.core.color}\nfill : bool or None\n    If True, fill in the area under univariate density curves or between\n    bivariate contours. If None, the default depends on ``multiple``.\n{params.dist.multiple}\ncommon_norm : bool\n    If True, scale each conditional density by the number of observations\n    such that the total area under all densities sums to 1. Otherwise,\n    normalize each density independently.\ncommon_grid : bool\n    If True, use the same evaluation grid for each kernel density estimate.\n    Only relevant with univariate data.\n{params.kde.cumulative}\n{params.kde.bw_method}\n{params.kde.bw_adjust}\nwarn_singular : bool\n    If True, issue a warning when trying to estimate the density of data\n    with zero variance.\n{params.dist.log_scale}\nlevels : int or vector\n    Number of contour levels or values to draw contours at. A vector argument\n    must have increasing values in [0, 1]. Levels correspond to iso-proportions\n    of the density: e.g., 20% of the probability mass will lie below the\n    contour drawn for 0.2. Only relevant with bivariate data.\nthresh : number in [0, 1]\n    Lowest iso-proportion level at which to draw a contour line. Ignored when\n    ``levels`` is a vector. Only relevant with bivariate data.\ngridsize : int\n    Number of points on each dimension of the evaluation grid.\n{params.kde.cut}\n{params.kde.clip}\n{params.dist.legend}\n{params.dist.cbar}\n{params.dist.cbar_ax}\n{params.dist.cbar_kws}\n{params.core.ax}\nkwargs\n    Other keyword arguments are passed to one of the following matplotlib\n    functions:\n\n    - :meth:`matplotlib.axes.Axes.plot` (univariate, ``fill=False``),\n    - :meth:`matplotlib.axes.Axes.fill_between` (univariate, ``fill=True``),\n    - :meth:`matplotlib.axes.Axes.contour` (bivariate, ``fill=False``),\n    - :meth:`matplotlib.axes.contourf` (bivariate, ``fill=True``).\n\nReturns\n-------\n{returns.ax}\n\nSee Also\n--------\n{seealso.displot}\n{seealso.histplot}\n{seealso.ecdfplot}\n{seealso.jointplot}\n{seealso.violinplot}\n\nNotes\n-----\n\nThe *bandwidth*, or standard deviation of the smoothing kernel, is an\nimportant parameter. Misspecification of the bandwidth can produce a\ndistorted representation of the data. Much like the choice of bin width in a\nhistogram, an over-smoothed curve can erase true features of a\ndistribution, while an under-smoothed curve can create false features out of\nrandom variability. The rule-of-thumb that sets the default bandwidth works\nbest when the true distribution is smooth, unimodal, and roughly bell-shaped.\nIt is always a good idea to check the default behavior by using ``bw_adjust``\nto increase or decrease the amount of smoothing.\n\nBecause the smoothing algorithm uses a Gaussian kernel, the estimated density\ncurve can extend to values that do not make sense for a particular dataset.\nFor example, the curve may be drawn over negative values when smoothing data\nthat are naturally positive. The ``cut`` and ``clip`` parameters can be used\nto control the extent of the curve, but datasets that have many observations\nclose to a natural boundary may be better served by a different visualization\nmethod.\n\nSimilar considerations apply when a dataset is naturally discrete or "spiky"\n(containing many repeated observations of the same value). Kernel density\nestimation will always produce a smooth curve, which would be misleading\nin these situations.\n\nThe units on the density axis are a common source of confusion. While kernel\ndensity estimation produces a probability distribution, the height of the curve\nat each point gives a density, not a probability. A probability can be obtained\nonly by integrating the density across a range. The curve is normalized so\nthat the integral over all possible values is 1, meaning that the scale of\nthe density axis depends on the data values.\n\nExamples\n--------\n\n.. include:: ../docstrings/kdeplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
ecdfplot.__doc__ = 'Plot empirical cumulative distribution functions.\n\nAn ECDF represents the proportion or count of observations falling below each\nunique value in a dataset. Compared to a histogram or density plot, it has the\nadvantage that each observation is visualized directly, meaning that there are\nno binning or smoothing parameters that need to be adjusted. It also aids direct\ncomparisons between multiple distributions. A downside is that the relationship\nbetween the appearance of the plot and the basic properties of the distribution\n(such as its central tendency, variance, and the presence of any bimodality)\nmay not be as intuitive.\n\nMore information is provided in the :ref:`user guide <tutorial_ecdf>`.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nweights : vector or key in ``data``\n    If provided, weight the contribution of the corresponding data points\n    towards the cumulative distribution using these values.\n{params.ecdf.stat}\n{params.ecdf.complementary}\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.dist.log_scale}\n{params.dist.legend}\n{params.core.ax}\nkwargs\n    Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.plot`.\n\nReturns\n-------\n{returns.ax}\n\nSee Also\n--------\n{seealso.displot}\n{seealso.histplot}\n{seealso.kdeplot}\n{seealso.rugplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/ecdfplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
rugplot.__doc__ = 'Plot marginal distributions by drawing ticks along the x and y axes.\n\nThis function is intended to complement other plots by showing the location\nof individual observations in an unobtrusive way.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nheight : float\n    Proportion of axes extent covered by each rug element. Can be negative.\nexpand_margins : bool\n    If True, increase the axes margins by the height of the rug to avoid\n    overlap with other elements.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\nlegend : bool\n    If False, do not add a legend for semantic variables.\n{params.core.ax}\nkwargs\n    Other keyword arguments are passed to\n    :meth:`matplotlib.collections.LineCollection`\n\nReturns\n-------\n{returns.ax}\n\nExamples\n--------\n\n.. include:: ../docstrings/rugplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'])
displot.__doc__ = 'Figure-level interface for drawing distribution plots onto a FacetGrid.\n\nThis function provides access to several approaches for visualizing the\nunivariate or bivariate distribution of data, including subsets of data\ndefined by semantic mapping and faceting across multiple subplots. The\n``kind`` parameter selects the approach to use:\n\n- :func:`histplot` (with ``kind="hist"``; the default)\n- :func:`kdeplot` (with ``kind="kde"``)\n- :func:`ecdfplot` (with ``kind="ecdf"``; univariate-only)\n\nAdditionally, a :func:`rugplot` can be added to any kind of plot to show\nindividual observations.\n\nExtra keyword arguments are passed to the underlying function, so you should\nrefer to the documentation for each to understand the complete set of options\nfor making plots with this interface.\n\nSee the :doc:`distribution plots tutorial <../tutorial/distributions>` for a more\nin-depth discussion of the relative strengths and weaknesses of each approach.\nThe distinction between figure-level and axes-level functions is explained\nfurther in the :doc:`user guide <../tutorial/function_overview>`.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\n{params.facets.rowcol}\nweights : vector or key in ``data``\n    Observation weights used for computing the distribution function.\nkind : {{"hist", "kde", "ecdf"}}\n    Approach for visualizing the data. Selects the underlying plotting function\n    and determines the additional set of valid parameters.\nrug : bool\n    If True, show each observation with marginal ticks (as in :func:`rugplot`).\nrug_kws : dict\n    Parameters to control the appearance of the rug plot.\n{params.dist.log_scale}\n{params.dist.legend}\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.core.color}\n{params.facets.col_wrap}\n{params.facets.rowcol_order}\n{params.facets.height}\n{params.facets.aspect}\n{params.facets.facet_kws}\nkwargs\n    Other keyword arguments are documented with the relevant axes-level function:\n\n    - :func:`histplot` (with ``kind="hist"``)\n    - :func:`kdeplot` (with ``kind="kde"``)\n    - :func:`ecdfplot` (with ``kind="ecdf"``)\n\nReturns\n-------\n{returns.facetgrid}\n\nSee Also\n--------\n{seealso.histplot}\n{seealso.kdeplot}\n{seealso.rugplot}\n{seealso.ecdfplot}\n{seealso.jointplot}\n\nExamples\n--------\n\nSee the API documentation for the axes-level functions for more details\nabout the breadth of options available for each plot kind.\n\n.. include:: ../docstrings/displot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])

def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * (np.percentile(a, 75) - np.percentile(a, 25))
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))

def distplot(a=None, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None, x=None):
    """
    DEPRECATED

    This function has been deprecated and will be removed in seaborn v0.14.0.
    It has been replaced by :func:`histplot` and :func:`displot`, two functions
    with a modern API and many more capabilities.

    For a guide to updating, please see this notebook:

    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

    """
    import warnings
    msg = (
        "distplot is a deprecated function and will be removed in seaborn v0.14.0.\n"
        "Please adapt your code to use either displot (a figure-level function with "
        "similar flexibility) or histplot (an axes-level function for histograms).\n"
        "For a guide to updating your code to use the new functions, please see "
        "https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751"
    )
    warnings.warn(msg, FutureWarning, stacklevel=2)

    if x is not None:
        a = x
    
    if ax is None:
        ax = plt.gca()
    
    if a is None:
        return ax

    a = np.asarray(a)
    if a.ndim > 1:
        a = a.squeeze()

    if bins is None:
        bins = min(_freedman_diaconis_bins(a), 50)

    if hist:
        if hist_kws is None:
            hist_kws = {}
        hist_kws.setdefault("alpha", 0.4)
        hist_kws.setdefault("density", norm_hist)

        orientation = "horizontal" if vertical else "vertical"
        ax.hist(a, bins=bins, orientation=orientation, color=color, **hist_kws)

    if kde:
        if kde_kws is None:
            kde_kws = {}
        kde_kws.setdefault("shade", True)

        kdeplot(a, vertical=vertical, ax=ax, color=color, **kde_kws)

    if rug:
        if rug_kws is None:
            rug_kws = {}
        rug_kws.setdefault("color", color)

        if vertical:
            ax.yaxis.tick_right()
        rugplot(a, vertical=vertical, ax=ax, **rug_kws)

    if fit is not None:
        if fit_kws is None:
            fit_kws = {}

        fitplot(a, fit, ax=ax, vertical=vertical, **fit_kws)

    if axlabel is not None:
        if vertical:
            ax.set_ylabel(axlabel)
        else:
            ax.set_xlabel(axlabel)

    if label is not None:
        if hist:
            ax.patches[0].set_label(label)
        elif kde:
            ax.lines[-1].set_label(label)
        elif rug:
            ax.lines[-1].set_label(label)

    return ax
