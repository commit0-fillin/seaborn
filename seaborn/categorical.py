from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import desaturate, _check_argument, _draw_figure, _default_color, _get_patch_legend_artist, _get_transform_functions, _scatter_legend_artist, _version_predates
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import EstimateAggregator, LetterValues, WeightedAggregator
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
__all__ = ['catplot', 'stripplot', 'swarmplot', 'boxplot', 'violinplot', 'boxenplot', 'pointplot', 'barplot', 'countplot']

class _CategoricalPlotter(VectorPlotter):
    wide_structure = {'x': '@columns', 'y': '@values', 'hue': '@columns'}
    flat_structure = {'y': '@values'}
    _legend_attributes = ['color']

    def __init__(self, data=None, variables={}, order=None, orient=None, require_numeric=False, color=None, legend='auto'):
        super().__init__(data=data, variables=variables)
        if self.input_format == 'wide' and orient in ['h', 'y']:
            self.plot_data = self.plot_data.rename(columns={'x': 'y', 'y': 'x'})
            orig_variables = set(self.variables)
            orig_x = self.variables.pop('x', None)
            orig_y = self.variables.pop('y', None)
            orig_x_type = self.var_types.pop('x', None)
            orig_y_type = self.var_types.pop('y', None)
            if 'x' in orig_variables:
                self.variables['y'] = orig_x
                self.var_types['y'] = orig_x_type
            if 'y' in orig_variables:
                self.variables['x'] = orig_y
                self.var_types['x'] = orig_y_type
        if self.input_format == 'wide' and 'hue' in self.variables and (color is not None):
            self.plot_data.drop('hue', axis=1)
            self.variables.pop('hue')
        self.orient = infer_orient(x=self.plot_data.get('x', None), y=self.plot_data.get('y', None), orient=orient, require_numeric=False)
        self.legend = legend
        if not self.has_xy_data:
            return
        if self.orient not in self.variables:
            self.variables[self.orient] = None
            self.var_types[self.orient] = 'categorical'
            self.plot_data[self.orient] = ''
        cat_levels = categorical_order(self.plot_data[self.orient], order)
        self.var_levels[self.orient] = cat_levels

    def _hue_backcompat(self, color, palette, hue_order, force_hue=False):
        """Implement backwards compatibility for hue parametrization.

        Note: the force_hue parameter is used so that functions can be shown to
        pass existing tests during refactoring and then tested for new behavior.
        It can be removed after completion of the work.

        """
        if force_hue or (palette is not None and self.hue_names is None):
            if isinstance(palette, dict):
                hue_order = sorted(palette)
            elif isinstance(palette, list):
                hue_order = palette
            elif isinstance(palette, str) and palette in SEABORN_PALETTES:
                hue_order = SEABORN_PALETTES[palette]
            
            if hue_order is not None:
                self.hue_names = hue_order
                self.n_colors = len(hue_order)
            
            if color is not None:
                self.colors = [color] * self.n_colors
            elif isinstance(palette, dict):
                self.colors = [palette[h] for h in hue_order]
            else:
                self.colors = color_palette(palette, self.n_colors)
        
        return color, palette, hue_order

    def _palette_without_hue_backcompat(self, palette, hue_order):
        """Provide one cycle where palette= implies hue= when not provided"""
        if palette is not None and self.hue_names is None:
            if isinstance(palette, dict):
                hue_order = sorted(palette)
            elif isinstance(palette, list):
                hue_order = palette
            elif isinstance(palette, str) and palette in SEABORN_PALETTES:
                hue_order = SEABORN_PALETTES[palette]
            
            if hue_order is not None:
                self.hue_names = hue_order
                self.n_colors = len(hue_order)
            
            if isinstance(palette, dict):
                self.colors = [palette[h] for h in hue_order]
            else:
                self.colors = color_palette(palette, self.n_colors)
        
        return palette, hue_order

    def _point_kwargs_backcompat(self, scale, join, kwargs):
        """Provide two cycles where scale= and join= work, but redirect to kwargs."""
        if scale is not None:
            kwargs.setdefault('scale', scale)
            warnings.warn("The `scale` parameter is deprecated. Use `kwargs={'scale': ...}` instead.",
                          FutureWarning)
        
        if join is not None:
            kwargs.setdefault('join', join)
            warnings.warn("The `join` parameter is deprecated. Use `kwargs={'join': ...}` instead.",
                          FutureWarning)
        
        return kwargs

    def _err_kws_backcompat(self, err_kws, errcolor, errwidth, capsize):
        """Provide two cycles where existing signature-level err_kws are handled."""
        err_kws = {} if err_kws is None else err_kws.copy()
        
        if errcolor is not None:
            err_kws.setdefault('color', errcolor)
            warnings.warn("The `errcolor` parameter is deprecated. Use `err_kws={'color': ...}` instead.",
                          FutureWarning)
        
        if errwidth is not None:
            err_kws.setdefault('linewidth', errwidth)
            warnings.warn("The `errwidth` parameter is deprecated. Use `err_kws={'linewidth': ...}` instead.",
                          FutureWarning)
        
        if capsize is not None:
            err_kws.setdefault('capsize', capsize)
            warnings.warn("The `capsize` parameter is deprecated. Use `err_kws={'capsize': ...}` instead.",
                          FutureWarning)
        
        return err_kws

    def _violin_scale_backcompat(self, scale, scale_hue, density_norm, common_norm):
        """Provide two cycles of backcompat for scale kwargs"""
        if scale is not None:
            density_norm = scale
            warnings.warn("The `scale` parameter is deprecated. Use `density_norm` instead.",
                          FutureWarning)
        
        if scale_hue is not None:
            common_norm = not scale_hue
            warnings.warn("The `scale_hue` parameter is deprecated. Use `common_norm` instead.",
                          FutureWarning)
        
        return density_norm, common_norm

    def _violin_bw_backcompat(self, bw, bw_method):
        """Provide two cycles of backcompat for violin bandwidth parameterization."""
        if bw is not None:
            if isinstance(bw, str):
                bw_method = bw
            else:
                bw_method = "scott" if bw == "scott" else bw
            warnings.warn("The `bw` parameter is deprecated. Use `bw_method` instead.",
                          FutureWarning)
        
        return bw_method

    def _boxen_scale_backcompat(self, scale, width_method):
        """Provide two cycles of backcompat for scale kwargs"""
        if scale is not None:
            width_method = scale
            warnings.warn("The `scale` parameter is deprecated. Use `width_method` instead.",
                          FutureWarning)
        
        return width_method

    def _complement_color(self, color, base_color, hue_map):
        """Allow a color to be set automatically using a basis of comparison."""
        if color is None:
            if base_color is None:
                color = "C0"
            else:
                color = set(hue_map.values()).difference({base_color}).pop()
        return color

    def _map_prop_with_hue(self, name, value, fallback, plot_kws):
        """Support pointplot behavior of modifying the marker/linestyle with hue."""
        if value is None:
            if self.hue_names is None:
                value = fallback
            else:
                value = [fallback] * len(self.hue_names)
        
        if isinstance(value, list):
            if len(value) != len(self.hue_names):
                raise ValueError(f"The `{name}` list must have {len(self.hue_names)} elements")
            plot_kws[name] = value[0]
        else:
            plot_kws[name] = value
        
        return value

    def _adjust_cat_axis(self, ax, axis):
        """Set ticks and limits for a categorical variable."""
        if axis == "x":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, len(self.plot_data[axis]) - .5, auto=None)
            ax.set_xticks(range(len(self.plot_data[axis])))
            ax.set_xticklabels(self.plot_data[axis])
        else:
            ax.yaxis.grid(False)
            ax.set_ylim(-.5, len(self.plot_data[axis]) - .5, auto=None)
            ax.set_yticks(range(len(self.plot_data[axis])))
            ax.set_yticklabels(self.plot_data[axis])

    def _dodge_needed(self):
        """Return True when use of `hue` would cause overlaps."""
        return self.hue_names is not None and self.dodge

    def _dodge(self, keys, data):
        """Apply a dodge transform to coordinates in place."""
        if not self._dodge_needed():
            return

        n_levels = len(self.hue_names)
        if self.dodge == "auto":
            dodge_width = .8 / n_levels
        else:
            dodge_width = self.dodge

        offsets = np.linspace(0, dodge_width, n_levels + 1)[:-1]
        offsets -= offsets.mean()

        for key in keys:
            data[key] = data[key] + offsets[data["_hue_order"]]

    def _invert_scale(self, ax, data, vars=('x', 'y')):
        """Undo scaling after computation so data are plotted correctly."""
        for var in vars:
            if var in data:
                scale = getattr(ax, f'get_{var}scale')()
                if scale == 'log':
                    data[var] = np.power(10, data[var])
                elif scale == 'symlog':
                    data[var] = np.sign(data[var]) * np.power(10, np.abs(data[var]))

    @property
    def _native_width(self):
        """Return unit of width separating categories on native numeric scale."""
        if self.orient == "x":
            return np.diff(self.ax.get_xticks()).min()
        else:
            return np.diff(self.ax.get_yticks()).min()

    def _nested_offsets(self, width, dodge):
        """Return offsets for each hue level for dodged plots."""
        if not self._dodge_needed():
            return [0]

        n_levels = len(self.hue_names)
        if dodge == "auto":
            dodge_width = width * .8 / n_levels
        else:
            dodge_width = dodge * width

        offsets = np.linspace(0, dodge_width, n_levels)
        offsets -= offsets.mean()

        return offsets

class _CategoricalAggPlotter(_CategoricalPlotter):
    flat_structure = {'x': '@index', 'y': '@values'}
_categorical_docs = dict(categorical_narrative=dedent('    See the :ref:`tutorial <categorical_tutorial>` for more information.\n\n    .. note::\n        By default, this function treats one of the variables as categorical\n        and draws data at ordinal positions (0, 1, ... n) on the relevant axis.\n        As of version 0.13.0, this can be disabled by setting `native_scale=True`.\n    '), input_params=dedent('    x, y, hue : names of variables in `data` or vector data\n        Inputs for plotting long-form data. See examples for interpretation.    '), categorical_data=dedent('    data : DataFrame, Series, dict, array, or list of arrays\n        Dataset for plotting. If `x` and `y` are absent, this is\n        interpreted as wide-form. Otherwise it is expected to be long-form.    '), order_vars=dedent('    order, hue_order : lists of strings\n        Order to plot the categorical levels in; otherwise the levels are\n        inferred from the data objects.    '), stat_api_params=dedent('    estimator : string or callable that maps vector -> scalar\n        Statistical function to estimate within each categorical bin.\n    errorbar : string, (string, number) tuple, callable or None\n        Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple\n        with a method name and a level parameter, or a function that maps from a\n        vector to a (min, max) interval, or None to hide errorbar. See the\n        :doc:`errorbar tutorial </tutorial/error_bars>` for more information.\n\n        .. versionadded:: v0.12.0\n    n_boot : int\n        Number of bootstrap samples used to compute confidence intervals.\n    seed : int, `numpy.random.Generator`, or `numpy.random.RandomState`\n        Seed or random number generator for reproducible bootstrapping.\n    units : name of variable in `data` or vector data\n        Identifier of sampling units; used by the errorbar function to\n        perform a multilevel bootstrap and account for repeated measures\n    weights : name of variable in `data` or vector data\n        Data values or column used to compute weighted statistics.\n        Note that the use of weights may limit other statistical options.\n\n        .. versionadded:: v0.13.1    '), ci=dedent('    ci : float\n        Level of the confidence interval to show, in [0, 100].\n\n        .. deprecated:: v0.12.0\n            Use `errorbar=("ci", ...)`.    '), orient=dedent('    orient : "v" | "h" | "x" | "y"\n        Orientation of the plot (vertical or horizontal). This is usually\n        inferred based on the type of the input variables, but it can be used\n        to resolve ambiguity when both `x` and `y` are numeric or when\n        plotting wide-form data.\n\n        .. versionchanged:: v0.13.0\n            Added \'x\'/\'y\' as options, equivalent to \'v\'/\'h\'.    '), color=dedent('    color : matplotlib color\n        Single color for the elements in the plot.    '), palette=dedent('    palette : palette name, list, dict, or :class:`matplotlib.colors.Colormap`\n        Color palette that maps the hue variable. If the palette is a dictionary,\n        keys should be names of levels and values should be matplotlib colors.\n        The type/value will sometimes force a qualitative/quantitative mapping.    '), hue_norm=dedent('    hue_norm : tuple or :class:`matplotlib.colors.Normalize` object\n        Normalization in data units for colormap applied to the `hue`\n        variable when it is numeric. Not relevant if `hue` is categorical.\n\n        .. versionadded:: v0.12.0    '), saturation=dedent('    saturation : float\n        Proportion of the original saturation to draw fill colors in. Large\n        patches often look better with desaturated colors, but set this to\n        `1` if you want the colors to perfectly match the input values.    '), capsize=dedent('    capsize : float\n        Width of the "caps" on error bars, relative to bar spacing.    '), errcolor=dedent("    errcolor : matplotlib color\n        Color used for the error bar lines.\n\n        .. deprecated:: 0.13.0\n            Use `err_kws={'color': ...}`.    "), errwidth=dedent("    errwidth : float\n        Thickness of error bar lines (and caps), in points.\n\n        .. deprecated:: 0.13.0\n            Use `err_kws={'linewidth': ...}`.    "), fill=dedent('    fill : bool\n        If True, use a solid patch. Otherwise, draw as line art.\n\n        .. versionadded:: v0.13.0    '), gap=dedent('    gap : float\n        Shrink on the orient axis by this factor to add a gap between dodged elements.\n\n        .. versionadded:: 0.13.0    '), width=dedent('    width : float\n        Width allotted to each element on the orient axis. When `native_scale=True`,\n        it is relative to the minimum distance between two values in the native scale.    '), dodge=dedent('    dodge : "auto" or bool\n        When hue mapping is used, whether elements should be narrowed and shifted along\n        the orient axis to eliminate overlap. If `"auto"`, set to `True` when the\n        orient variable is crossed with the categorical variable or `False` otherwise.\n\n        .. versionchanged:: 0.13.0\n\n            Added `"auto"` mode as a new default.    '), linewidth=dedent('    linewidth : float\n        Width of the lines that frame the plot elements.    '), linecolor=dedent('    linecolor : color\n        Color to use for line elements, when `fill` is True.\n\n        .. versionadded:: v0.13.0    '), log_scale=dedent('    log_scale : bool or number, or pair of bools or numbers\n        Set axis scale(s) to log. A single value sets the data axis for any numeric\n        axes in the plot. A pair of values sets each axis independently.\n        Numeric values are interpreted as the desired base (default 10).\n        When `None` or `False`, seaborn defers to the existing Axes scale.\n\n        .. versionadded:: v0.13.0    '), native_scale=dedent('    native_scale : bool\n        When True, numeric or datetime values on the categorical axis will maintain\n        their original scaling rather than being converted to fixed indices.\n\n        .. versionadded:: v0.13.0    '), formatter=dedent('    formatter : callable\n        Function for converting categorical data into strings. Affects both grouping\n        and tick labels.\n\n        .. versionadded:: v0.13.0    '), legend=dedent('    legend : "auto", "brief", "full", or False\n        How to draw the legend. If "brief", numeric `hue` and `size`\n        variables will be represented with a sample of evenly spaced values.\n        If "full", every group will get an entry in the legend. If "auto",\n        choose between brief or full representation based on number of levels.\n        If `False`, no legend data is added and no legend is drawn.\n\n        .. versionadded:: v0.13.0    '), err_kws=dedent('    err_kws : dict\n        Parameters of :class:`matplotlib.lines.Line2D`, for the error bar artists.\n\n        .. versionadded:: v0.13.0    '), ax_in=dedent('    ax : matplotlib Axes\n        Axes object to draw the plot onto, otherwise uses the current Axes.    '), ax_out=dedent('    ax : matplotlib Axes\n        Returns the Axes object with the plot drawn onto it.    '), boxplot=dedent('    boxplot : A traditional box-and-whisker plot with a similar API.    '), violinplot=dedent('    violinplot : A combination of boxplot and kernel density estimation.    '), stripplot=dedent('    stripplot : A scatterplot where one variable is categorical. Can be used\n                in conjunction with other plots to show each observation.    '), swarmplot=dedent('    swarmplot : A categorical scatterplot where the points do not overlap. Can\n                be used with other plots to show each observation.    '), barplot=dedent('    barplot : Show point estimates and confidence intervals using bars.    '), countplot=dedent('    countplot : Show the counts of observations in each categorical bin.    '), pointplot=dedent('    pointplot : Show point estimates and confidence intervals using dots.    '), catplot=dedent('    catplot : Combine a categorical plot with a :class:`FacetGrid`.    '), boxenplot=dedent('    boxenplot : An enhanced boxplot for larger datasets.    '))
_categorical_docs.update(_facet_docs)
boxplot.__doc__ = dedent('    Draw a box plot to show distributions with respect to categories.\n\n    A box plot (or box-and-whisker plot) shows the distribution of quantitative\n    data in a way that facilitates comparisons between variables or across\n    levels of a categorical variable. The box shows the quartiles of the\n    dataset while the whiskers extend to show the rest of the distribution,\n    except for points that are determined to be "outliers" using a method\n    that is a function of the inter-quartile range.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {orient}\n    {color}\n    {palette}\n    {saturation}\n    {fill}\n    {dodge}\n    {width}\n    {gap}\n    whis : float or pair of floats\n        Paramater that controls whisker length. If scalar, whiskers are drawn\n        to the farthest datapoint within *whis * IQR* from the nearest hinge.\n        If a tuple, it is interpreted as percentiles that whiskers represent.\n    {linecolor}\n    {linewidth}\n    fliersize : float\n        Size of the markers used to indicate outlier observations.\n    {hue_norm}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    {ax_in}\n    kwargs : key, value mappings\n        Other keyword arguments are passed through to\n        :meth:`matplotlib.axes.Axes.boxplot`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {violinplot}\n    {stripplot}\n    {swarmplot}\n    {catplot}\n\n    Examples\n    --------\n    .. include:: ../docstrings/boxplot.rst\n\n    ').format(**_categorical_docs)
violinplot.__doc__ = dedent('    Draw a patch representing a KDE and add observations or box plot statistics.\n\n    A violin plot plays a similar role as a box-and-whisker plot. It shows the\n    distribution of data points after grouping by one (or more) variables.\n    Unlike a box plot, each violin is drawn using a kernel density estimate\n    of the underlying distribution.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {orient}\n    {color}\n    {palette}\n    {saturation}\n    {fill}\n    inner : {{"box", "quart", "point", "stick", None}}\n        Representation of the data in the violin interior. One of the following:\n\n        - `"box"`: draw a miniature box-and-whisker plot\n        - `"quart"`: show the quartiles of the data\n        - `"point"` or `"stick"`: show each observation\n    split : bool\n        Show an un-mirrored distribution, alternating sides when using `hue`.\n\n        .. versionchanged:: v0.13.0\n            Previously, this option required a `hue` variable with exactly two levels.\n    {width}\n    {dodge}\n    {gap}\n    {linewidth}\n    {linecolor}\n    cut : float\n        Distance, in units of bandwidth, to extend the density past extreme\n        datapoints. Set to 0 to limit the violin within the data range.\n    gridsize : int\n        Number of points in the discrete grid used to evaluate the KDE.\n    bw_method : {{"scott", "silverman", float}}\n        Either the name of a reference rule or the scale factor to use when\n        computing the kernel bandwidth. The actual kernel size will be\n        determined by multiplying the scale factor by the standard deviation of\n        the data within each group.\n\n        .. versionadded:: v0.13.0\n    bw_adjust: float\n        Factor that scales the bandwidth to use more or less smoothing.\n\n        .. versionadded:: v0.13.0\n    density_norm : {{"area", "count", "width"}}\n        Method that normalizes each density to determine the violin\'s width.\n        If `area`, each violin will have the same area. If `count`, the width\n        will be proportional to the number of observations. If `width`, each\n        violin will have the same width.\n\n        .. versionadded:: v0.13.0\n    common_norm : bool\n        When `True`, normalize the density across all violins.\n\n        .. versionadded:: v0.13.0\n    {hue_norm}\n    {formatter}\n    {log_scale}\n    {native_scale}\n    {legend}\n    scale : {{"area", "count", "width"}}\n        .. deprecated:: v0.13.0\n            See `density_norm`.\n    scale_hue : bool\n        .. deprecated:: v0.13.0\n            See `common_norm`.\n    bw : {{\'scott\', \'silverman\', float}}\n        .. deprecated:: v0.13.0\n            See `bw_method` and `bw_adjust`.\n    inner_kws : dict of key, value mappings\n        Keyword arguments for the "inner" plot, passed to one of:\n\n        - :class:`matplotlib.collections.LineCollection` (with `inner="stick"`)\n        - :meth:`matplotlib.axes.Axes.scatter` (with `inner="point"`)\n        - :meth:`matplotlib.axes.Axes.plot` (with `inner="quart"` or `inner="box"`)\n\n        Additionally, with `inner="box"`, the keywords `box_width`, `whis_width`,\n        and `marker` receive special handling for the components of the "box" plot.\n\n        .. versionadded:: v0.13.0\n    {ax_in}\n    kwargs : key, value mappings\n        Keyword arguments for the violin patches, passsed through to\n        :meth:`matplotlib.axes.Axes.fill_between`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {boxplot}\n    {stripplot}\n    {swarmplot}\n    {catplot}\n\n    Examples\n    --------\n    .. include:: ../docstrings/violinplot.rst\n\n    ').format(**_categorical_docs)
boxenplot.__doc__ = dedent('    Draw an enhanced box plot for larger datasets.\n\n    This style of plot was originally named a "letter value" plot because it\n    shows a large number of quantiles that are defined as "letter values".  It\n    is similar to a box plot in plotting a nonparametric representation of a\n    distribution in which all features correspond to actual observations. By\n    plotting more quantiles, it provides more information about the shape of\n    the distribution, particularly in the tails.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {orient}\n    {color}\n    {palette}\n    {saturation}\n    {fill}\n    {dodge}\n    {width}\n    {gap}\n    {linewidth}\n    {linecolor}\n    width_method : {{"exponential", "linear", "area"}}\n        Method to use for the width of the letter value boxes:\n\n        - `"exponential"`: Represent the corresponding percentile\n        - `"linear"`: Decrease by a constant amount for each box\n        - `"area"`: Represent the density of data points in that box\n    k_depth : {{"tukey", "proportion", "trustworthy", "full"}} or int\n        The number of levels to compute and draw in each tail:\n\n        - `"tukey"`: Use log2(n) - 3 levels, covering similar range as boxplot whiskers\n        - `"proportion"`: Leave approximately `outlier_prop` fliers\n        - `"trusthworthy"`: Extend to level with confidence of at least `trust_alpha`\n        - `"full"`: Use log2(n) + 1 levels and extend to most extreme points\n    outlier_prop : float\n        Proportion of data expected to be outliers; used when `k_depth="proportion"`.\n    trust_alpha : float\n        Confidence threshold for most extreme level; used when `k_depth="trustworthy"`.\n    showfliers : bool\n        If False, suppress the plotting of outliers.\n    {hue_norm}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    box_kws: dict\n        Keyword arguments for the box artists; passed to\n        :class:`matplotlib.patches.Rectangle`.\n\n        .. versionadded:: v0.12.0\n    line_kws: dict\n        Keyword arguments for the line denoting the median; passed to\n        :meth:`matplotlib.axes.Axes.plot`.\n\n        .. versionadded:: v0.12.0\n    flier_kws: dict\n        Keyword arguments for the scatter denoting the outlier observations;\n        passed to :meth:`matplotlib.axes.Axes.scatter`.\n\n        .. versionadded:: v0.12.0\n    {ax_in}\n    kwargs : key, value mappings\n        Other keyword arguments are passed to :class:`matplotlib.patches.Rectangle`,\n        superceded by those in `box_kws`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {violinplot}\n    {boxplot}\n    {catplot}\n\n    Notes\n    -----\n\n    For a more extensive explanation, you can read the paper that introduced the plot:\n    https://vita.had.co.nz/papers/letter-value-plot.html\n\n    Examples\n    --------\n    .. include:: ../docstrings/boxenplot.rst\n\n    ').format(**_categorical_docs)
stripplot.__doc__ = dedent('    Draw a categorical scatterplot using jitter to reduce overplotting.\n\n    A strip plot can be drawn on its own, but it is also a good complement\n    to a box or violin plot in cases where you want to show all observations\n    along with some representation of the underlying distribution.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    jitter : float, `True`/`1` is special-cased\n        Amount of jitter (only along the categorical axis) to apply. This\n        can be useful when you have many points and they overlap, so that\n        it is easier to see the distribution. You can specify the amount\n        of jitter (half the width of the uniform random variable support),\n        or use `True` for a good default.\n    dodge : bool\n        When a `hue` variable is assigned, setting this to `True` will\n        separate the strips for different hue levels along the categorical\n        axis and narrow the amount of space allotedto each strip. Otherwise,\n        the points for each level will be plotted in the same strip.\n    {orient}\n    {color}\n    {palette}\n    size : float\n        Radius of the markers, in points.\n    edgecolor : matplotlib color, "gray" is special-cased\n        Color of the lines around each point. If you pass `"gray"`, the\n        brightness is determined by the color palette used for the body\n        of the points. Note that `stripplot` has `linewidth=0` by default,\n        so edge colors are only visible with nonzero line width.\n    {linewidth}\n    {hue_norm}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    {ax_in}\n    kwargs : key, value mappings\n        Other keyword arguments are passed through to\n        :meth:`matplotlib.axes.Axes.scatter`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {swarmplot}\n    {boxplot}\n    {violinplot}\n    {catplot}\n\n    Examples\n    --------\n    .. include:: ../docstrings/stripplot.rst\n\n    ').format(**_categorical_docs)
swarmplot.__doc__ = dedent('    Draw a categorical scatterplot with points adjusted to be non-overlapping.\n\n    This function is similar to :func:`stripplot`, but the points are adjusted\n    (only along the categorical axis) so that they don\'t overlap. This gives a\n    better representation of the distribution of values, but it does not scale\n    well to large numbers of observations. This style of plot is sometimes\n    called a "beeswarm".\n\n    A swarm plot can be drawn on its own, but it is also a good complement\n    to a box or violin plot in cases where you want to show all observations\n    along with some representation of the underlying distribution.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    dodge : bool\n        When a `hue` variable is assigned, setting this to `True` will\n        separate the swarms for different hue levels along the categorical\n        axis and narrow the amount of space allotedto each strip. Otherwise,\n        the points for each level will be plotted in the same swarm.\n    {orient}\n    {color}\n    {palette}\n    size : float\n        Radius of the markers, in points.\n    edgecolor : matplotlib color, "gray" is special-cased\n        Color of the lines around each point. If you pass `"gray"`, the\n        brightness is determined by the color palette used for the body\n        of the points.\n    {linewidth}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    {ax_in}\n    kwargs : key, value mappings\n        Other keyword arguments are passed through to\n        :meth:`matplotlib.axes.Axes.scatter`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {boxplot}\n    {violinplot}\n    {stripplot}\n    {catplot}\n\n    Examples\n    --------\n    .. include:: ../docstrings/swarmplot.rst\n\n    ').format(**_categorical_docs)
barplot.__doc__ = dedent('    Show point estimates and errors as rectangular bars.\n\n    A bar plot represents an aggregate or statistical estimate for a numeric\n    variable with the height of each rectangle and indicates the uncertainty\n    around that estimate using an error bar. Bar plots include 0 in the\n    axis range, and they are a good choice when 0 is a meaningful value\n    for the variable to take.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {stat_api_params}\n    {orient}\n    {color}\n    {palette}\n    {saturation}\n    {fill}\n    {hue_norm}\n    {width}\n    {dodge}\n    {gap}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    {capsize}\n    {err_kws}\n    {ci}\n    {errcolor}\n    {errwidth}\n    {ax_in}\n    kwargs : key, value mappings\n        Other parameters are passed through to :class:`matplotlib.patches.Rectangle`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {countplot}\n    {pointplot}\n    {catplot}\n\n    Notes\n    -----\n\n    For datasets where 0 is not a meaningful value, a :func:`pointplot` will\n    allow you to focus on differences between levels of one or more categorical\n    variables.\n\n    It is also important to keep in mind that a bar plot shows only the mean (or\n    other aggregate) value, but it is often more informative to show the\n    distribution of values at each level of the categorical variables. In those\n    cases, approaches such as a :func:`boxplot` or :func:`violinplot` may be\n    more appropriate.\n\n    Examples\n    --------\n    .. include:: ../docstrings/barplot.rst\n\n    ').format(**_categorical_docs)
pointplot.__doc__ = dedent('    Show point estimates and errors using lines with markers.\n\n    A point plot represents an estimate of central tendency for a numeric\n    variable by the position of the dot and provides some indication of the\n    uncertainty around that estimate using error bars.\n\n    Point plots can be more useful than bar plots for focusing comparisons\n    between different levels of one or more categorical variables. They are\n    particularly adept at showing interactions: how the relationship between\n    levels of one categorical variable changes across levels of a second\n    categorical variable. The lines that join each point from the same `hue`\n    level allow interactions to be judged by differences in slope, which is\n    easier for the eyes than comparing the heights of several groups of points\n    or bars.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {stat_api_params}\n    {color}\n    {palette}\n    markers : string or list of strings\n        Markers to use for each of the `hue` levels.\n    linestyles : string or list of strings\n        Line styles to use for each of the `hue` levels.\n    dodge : bool or float\n        Amount to separate the points for each level of the `hue` variable along\n        the categorical axis. Setting to `True` will apply a small default.\n    {log_scale}\n    {native_scale}\n    {orient}\n    {capsize}\n    {formatter}\n    {legend}\n    {err_kws}\n    {ci}\n    {errwidth}\n    join : bool\n        If `True`, connect point estimates with a line.\n\n        .. deprecated:: v0.13.0\n            Set `linestyle="none"` to remove the lines between the points.\n    scale : float\n        Scale factor for the plot elements.\n\n        .. deprecated:: v0.13.0\n            Control element sizes with :class:`matplotlib.lines.Line2D` parameters.\n    {ax_in}\n    kwargs : key, value mappings\n        Other parameters are passed through to :class:`matplotlib.lines.Line2D`.\n\n        .. versionadded:: v0.13.0\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    {barplot}\n    {catplot}\n\n    Notes\n    -----\n    It is important to keep in mind that a point plot shows only the mean (or\n    other estimator) value, but in many cases it may be more informative to\n    show the distribution of values at each level of the categorical variables.\n    In that case, other approaches such as a box or violin plot may be more\n    appropriate.\n\n    Examples\n    --------\n    .. include:: ../docstrings/pointplot.rst\n\n    ').format(**_categorical_docs)
countplot.__doc__ = dedent("    Show the counts of observations in each categorical bin using bars.\n\n    A count plot can be thought of as a histogram across a categorical, instead\n    of quantitative, variable. The basic API and options are identical to those\n    for :func:`barplot`, so you can compare counts across nested variables.\n\n    Note that :func:`histplot` function offers similar functionality with additional\n    features (e.g. bar stacking), although its default behavior is somewhat different.\n\n    {categorical_narrative}\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    {order_vars}\n    {orient}\n    {color}\n    {palette}\n    {saturation}\n    {hue_norm}\n    stat : {{'count', 'percent', 'proportion', 'probability'}}\n        Statistic to compute; when not `'count'`, bar heights will be normalized so that\n        they sum to 100 (for `'percent'`) or 1 (otherwise) across the plot.\n\n        .. versionadded:: v0.13.0\n    {width}\n    {dodge}\n    {log_scale}\n    {native_scale}\n    {formatter}\n    {legend}\n    {ax_in}\n    kwargs : key, value mappings\n        Other parameters are passed through to :class:`matplotlib.patches.Rectangle`.\n\n    Returns\n    -------\n    {ax_out}\n\n    See Also\n    --------\n    histplot : Bin and count observations with additional options.\n    {barplot}\n    {catplot}\n\n    Examples\n    --------\n    .. include:: ../docstrings/countplot.rst\n\n    ").format(**_categorical_docs)
catplot.__doc__ = dedent('    Figure-level interface for drawing categorical plots onto a FacetGrid.\n\n    This function provides access to several axes-level functions that\n    show the relationship between a numerical and one or more categorical\n    variables using one of several visual representations. The `kind`\n    parameter selects the underlying axes-level function to use.\n\n    Categorical scatterplots:\n\n    - :func:`stripplot` (with `kind="strip"`; the default)\n    - :func:`swarmplot` (with `kind="swarm"`)\n\n    Categorical distribution plots:\n\n    - :func:`boxplot` (with `kind="box"`)\n    - :func:`violinplot` (with `kind="violin"`)\n    - :func:`boxenplot` (with `kind="boxen"`)\n\n    Categorical estimate plots:\n\n    - :func:`pointplot` (with `kind="point"`)\n    - :func:`barplot` (with `kind="bar"`)\n    - :func:`countplot` (with `kind="count"`)\n\n    Extra keyword arguments are passed to the underlying function, so you\n    should refer to the documentation for each to see kind-specific options.\n\n    {categorical_narrative}\n\n    After plotting, the :class:`FacetGrid` with the plot is returned and can\n    be used directly to tweak supporting plot details or add other layers.\n\n    Parameters\n    ----------\n    {categorical_data}\n    {input_params}\n    row, col : names of variables in `data` or vector data\n        Categorical variables that will determine the faceting of the grid.\n    kind : str\n        The kind of plot to draw, corresponds to the name of a categorical\n        axes-level plotting function. Options are: "strip", "swarm", "box", "violin",\n        "boxen", "point", "bar", or "count".\n    {stat_api_params}\n    {order_vars}\n    row_order, col_order : lists of strings\n        Order to organize the rows and/or columns of the grid in; otherwise the\n        orders are inferred from the data objects.\n    {col_wrap}\n    {height}\n    {aspect}\n    {native_scale}\n    {formatter}\n    {orient}\n    {color}\n    {palette}\n    {hue_norm}\n    {legend}\n    {legend_out}\n    {share_xy}\n    {margin_titles}\n    facet_kws : dict\n        Dictionary of other keyword arguments to pass to :class:`FacetGrid`.\n    kwargs : key, value pairings\n        Other keyword arguments are passed through to the underlying plotting\n        function.\n\n    Returns\n    -------\n    :class:`FacetGrid`\n        Returns the :class:`FacetGrid` object with the plot on it for further\n        tweaking.\n\n    Examples\n    --------\n    .. include:: ../docstrings/catplot.rst\n\n    ').format(**_categorical_docs)

class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""

    def __init__(self, orient='x', width=0.8, warn_thresh=0.05):
        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points, center):
        """Swarm `points`, a PathCollection, around the `center` position."""
        ax = points.axes
        dpi = ax.figure.dpi
        orig_xy_data = points.get_offsets()
        cat_idx = 1 if self.orient == 'y' else 0
        orig_xy_data[:, cat_idx] = center
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)
        if self.orient == 'y':
            orig_xy = orig_xy[:, [1, 0]]
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)
        if self.orient == 'y':
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T
        t_fwd, t_inv = _get_transform_functions(ax, self.orient)
        if self.orient == 'y':
            self.add_gutters(new_y_data, center, t_fwd, t_inv)
        else:
            self.add_gutters(new_x_data, center, t_fwd, t_inv)
        if self.orient == 'y':
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        pass

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        pass

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        pass

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""
        pass

    def add_gutters(self, points, center, trans_fwd, trans_inv):
        """Stop points from extending beyond their territory."""
        pass
BoxPlotArtists = namedtuple('BoxPlotArtists', 'box median whiskers caps fliers mean')

class BoxPlotContainer:

    def __init__(self, artist_dict):
        self.boxes = artist_dict['boxes']
        self.medians = artist_dict['medians']
        self.whiskers = artist_dict['whiskers']
        self.caps = artist_dict['caps']
        self.fliers = artist_dict['fliers']
        self.means = artist_dict['means']
        self._label = None
        self._children = [*self.boxes, *self.medians, *self.whiskers, *self.caps, *self.fliers, *self.means]

    def __repr__(self):
        return f'<BoxPlotContainer object with {len(self.boxes)} boxes>'

    def __getitem__(self, idx):
        pair_slice = slice(2 * idx, 2 * idx + 2)
        return BoxPlotArtists(self.boxes[idx] if self.boxes else [], self.medians[idx] if self.medians else [], self.whiskers[pair_slice] if self.whiskers else [], self.caps[pair_slice] if self.caps else [], self.fliers[idx] if self.fliers else [], self.means[idx] if self.means else [])

    def __iter__(self):
        yield from (self[i] for i in range(len(self.boxes)))
