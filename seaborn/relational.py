from functools import partial
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from ._base import VectorPlotter
from .utils import adjust_legend_subtitles, _default_color, _deprecate_ci, _get_transform_functions, _scatter_legend_artist
from ._compat import groupby_apply_include_groups
from ._statistics import EstimateAggregator, WeightedAggregator
from .axisgrid import FacetGrid, _facet_docs
from ._docstrings import DocstringComponents, _core_docs
__all__ = ['relplot', 'scatterplot', 'lineplot']
_relational_narrative = DocstringComponents(dict(main_api='\nThe relationship between `x` and `y` can be shown for different subsets\nof the data using the `hue`, `size`, and `style` parameters. These\nparameters control what visual semantics are used to identify the different\nsubsets. It is possible to show up to three dimensions independently by\nusing all three semantic types, but this style of plot can be hard to\ninterpret and is often ineffective. Using redundant semantics (i.e. both\n`hue` and `style` for the same variable) can be helpful for making\ngraphics more accessible.\n\nSee the :ref:`tutorial <relational_tutorial>` for more information.\n    ', relational_semantic='\nThe default treatment of the `hue` (and to a lesser extent, `size`)\nsemantic, if present, depends on whether the variable is inferred to\nrepresent "numeric" or "categorical" data. In particular, numeric variables\nare represented with a sequential colormap by default, and the legend\nentries show regular "ticks" with values that may or may not exist in the\ndata. This behavior can be controlled through various parameters, as\ndescribed and illustrated below.\n    '))
_relational_docs = dict(data_vars='\nx, y : names of variables in `data` or vector data\n    Input data variables; must be numeric. Can pass data directly or\n    reference columns in `data`.\n    ', data='\ndata : DataFrame, array, or list of arrays\n    Input data structure. If `x` and `y` are specified as names, this\n    should be a "long-form" DataFrame containing those columns. Otherwise\n    it is treated as "wide-form" data and grouping variables are ignored.\n    See the examples for the various ways this parameter can be specified\n    and the different effects of each.\n    ', palette='\npalette : string, list, dict, or matplotlib colormap\n    An object that determines how colors are chosen when `hue` is used.\n    It can be the name of a seaborn palette or matplotlib colormap, a list\n    of colors (anything matplotlib understands), a dict mapping levels\n    of the `hue` variable to colors, or a matplotlib colormap object.\n    ', hue_order='\nhue_order : list\n    Specified order for the appearance of the `hue` variable levels,\n    otherwise they are determined from the data. Not relevant when the\n    `hue` variable is numeric.\n    ', hue_norm='\nhue_norm : tuple or :class:`matplotlib.colors.Normalize` object\n    Normalization in data units for colormap applied to the `hue`\n    variable when it is numeric. Not relevant if `hue` is categorical.\n    ', sizes='\nsizes : list, dict, or tuple\n    An object that determines how sizes are chosen when `size` is used.\n    List or dict arguments should provide a size for each unique data value,\n    which forces a categorical interpretation. The argument may also be a\n    min, max tuple.\n    ', size_order='\nsize_order : list\n    Specified order for appearance of the `size` variable levels,\n    otherwise they are determined from the data. Not relevant when the\n    `size` variable is numeric.\n    ', size_norm='\nsize_norm : tuple or Normalize object\n    Normalization in data units for scaling plot objects when the\n    `size` variable is numeric.\n    ', dashes='\ndashes : boolean, list, or dictionary\n    Object determining how to draw the lines for different levels of the\n    `style` variable. Setting to `True` will use default dash codes, or\n    you can pass a list of dash codes or a dictionary mapping levels of the\n    `style` variable to dash codes. Setting to `False` will use solid\n    lines for all subsets. Dashes are specified as in matplotlib: a tuple\n    of `(segment, gap)` lengths, or an empty string to draw a solid line.\n    ', markers='\nmarkers : boolean, list, or dictionary\n    Object determining how to draw the markers for different levels of the\n    `style` variable. Setting to `True` will use default markers, or\n    you can pass a list of markers or a dictionary mapping levels of the\n    `style` variable to markers. Setting to `False` will draw\n    marker-less lines.  Markers are specified as in matplotlib.\n    ', style_order='\nstyle_order : list\n    Specified order for appearance of the `style` variable levels\n    otherwise they are determined from the data. Not relevant when the\n    `style` variable is numeric.\n    ', units='\nunits : vector or key in `data`\n    Grouping variable identifying sampling units. When used, a separate\n    line will be drawn for each unit with appropriate semantics, but no\n    legend entry will be added. Useful for showing distribution of\n    experimental replicates when exact identities are not needed.\n    ', estimator='\nestimator : name of pandas method or callable or None\n    Method for aggregating across multiple observations of the `y`\n    variable at the same `x` level. If `None`, all observations will\n    be drawn.\n    ', ci='\nci : int or "sd" or None\n    Size of the confidence interval to draw when aggregating.\n\n    .. deprecated:: 0.12.0\n        Use the new `errorbar` parameter for more flexibility.\n\n    ', n_boot='\nn_boot : int\n    Number of bootstraps to use for computing the confidence interval.\n    ', seed='\nseed : int, numpy.random.Generator, or numpy.random.RandomState\n    Seed or random number generator for reproducible bootstrapping.\n    ', legend='\nlegend : "auto", "brief", "full", or False\n    How to draw the legend. If "brief", numeric `hue` and `size`\n    variables will be represented with a sample of evenly spaced values.\n    If "full", every group will get an entry in the legend. If "auto",\n    choose between brief or full representation based on number of levels.\n    If `False`, no legend data is added and no legend is drawn.\n    ', ax_in='\nax : matplotlib Axes\n    Axes object to draw the plot onto, otherwise uses the current Axes.\n    ', ax_out='\nax : matplotlib Axes\n    Returns the Axes object with the plot drawn onto it.\n    ')
_param_docs = DocstringComponents.from_nested_components(core=_core_docs['params'], facets=DocstringComponents(_facet_docs), rel=DocstringComponents(_relational_docs), stat=DocstringComponents.from_function_params(EstimateAggregator.__init__))

class _RelationalPlotter(VectorPlotter):
    wide_structure = {'x': '@index', 'y': '@values', 'hue': '@columns', 'style': '@columns'}
    sort = True

class _LinePlotter(_RelationalPlotter):
    _legend_attributes = ['color', 'linewidth', 'marker', 'dashes']

    def __init__(self, *, data=None, variables={}, estimator=None, n_boot=None, seed=None, errorbar=None, sort=True, orient='x', err_style=None, err_kws=None, legend=None):
        self._default_size_range = np.r_[0.5, 2] * mpl.rcParams['lines.linewidth']
        super().__init__(data=data, variables=variables)
        self.estimator = estimator if estimator is not None else np.mean
        self.errorbar = errorbar
        self.n_boot = n_boot if n_boot is not None else 1000
        self.seed = seed
        self.sort = sort
        self.orient = orient
        self.err_style = err_style if err_style is not None else 'band'
        self.err_kws = {} if err_kws is None else err_kws
        self.legend = legend

        # Initialize additional attributes
        self.x = None
        self.y = None
        self.x_label = None
        self.y_label = None
        self.colors = []

        # Process the input data
        self._process_data()

    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""
        # Set default values for matplotlib parameters
        defaults = {
            "linewidth": mpl.rcParams["lines.linewidth"],
            "color": self.colors[0] if self.colors else None,
            "label": "",
            "zorder": 3,
        }
        
        # Update defaults with provided kwargs
        for key, val in defaults.items():
            kws.setdefault(key, val)

        # Plot the data
        if self.orient == "y":
            ax.plot(self.y, self.x, **kws)
        else:
            ax.plot(self.x, self.y, **kws)

        # Set labels if provided
        if self.x_label:
            ax.set_xlabel(self.x_label)
        if self.y_label:
            ax.set_ylabel(self.y_label)

        return ax

class _ScatterPlotter(_RelationalPlotter):
    _legend_attributes = ['color', 's', 'marker']

    def __init__(self, *, data=None, variables={}, legend=None):
        self._default_size_range = np.r_[0.5, 2] * np.square(mpl.rcParams['lines.markersize'])
        super().__init__(data=data, variables=variables)
        self.legend = legend
lineplot.__doc__ = 'Draw a line plot with possibility of several semantic groupings.\n\n{narrative.main_api}\n\n{narrative.relational_semantic}\n\nBy default, the plot aggregates over multiple `y` values at each value of\n`x` and shows an estimate of the central tendency and a confidence\ninterval for that estimate.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\nhue : vector or key in `data`\n    Grouping variable that will produce lines with different colors.\n    Can be either categorical or numeric, although color mapping will\n    behave differently in latter case.\nsize : vector or key in `data`\n    Grouping variable that will produce lines with different widths.\n    Can be either categorical or numeric, although size mapping will\n    behave differently in latter case.\nstyle : vector or key in `data`\n    Grouping variable that will produce lines with different dashes\n    and/or markers. Can have a numeric dtype but will always be treated\n    as categorical.\n{params.rel.units}\nweights : vector or key in `data`\n    Data values or column used to compute weighted estimation.\n    Note that use of weights currently limits the choice of statistics\n    to a \'mean\' estimator and \'ci\' errorbar.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.rel.sizes}\n{params.rel.size_order}\n{params.rel.size_norm}\n{params.rel.dashes}\n{params.rel.markers}\n{params.rel.style_order}\n{params.rel.estimator}\n{params.stat.errorbar}\n{params.rel.n_boot}\n{params.rel.seed}\norient : "x" or "y"\n    Dimension along which the data are sorted / aggregated. Equivalently,\n    the "independent variable" of the resulting function.\nsort : boolean\n    If True, the data will be sorted by the x and y variables, otherwise\n    lines will connect points in the order they appear in the dataset.\nerr_style : "band" or "bars"\n    Whether to draw the confidence intervals with translucent error bands\n    or discrete error bars.\nerr_kws : dict of keyword arguments\n    Additional parameters to control the aesthetics of the error bars. The\n    kwargs are passed either to :meth:`matplotlib.axes.Axes.fill_between`\n    or :meth:`matplotlib.axes.Axes.errorbar`, depending on `err_style`.\n{params.rel.legend}\n{params.rel.ci}\n{params.core.ax}\nkwargs : key, value mappings\n    Other keyword arguments are passed down to\n    :meth:`matplotlib.axes.Axes.plot`.\n\nReturns\n-------\n{returns.ax}\n\nSee Also\n--------\n{seealso.scatterplot}\n{seealso.pointplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/lineplot.rst\n\n'.format(narrative=_relational_narrative, params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
scatterplot.__doc__ = 'Draw a scatter plot with possibility of several semantic groupings.\n\n{narrative.main_api}\n\n{narrative.relational_semantic}\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\nhue : vector or key in `data`\n    Grouping variable that will produce points with different colors.\n    Can be either categorical or numeric, although color mapping will\n    behave differently in latter case.\nsize : vector or key in `data`\n    Grouping variable that will produce points with different sizes.\n    Can be either categorical or numeric, although size mapping will\n    behave differently in latter case.\nstyle : vector or key in `data`\n    Grouping variable that will produce points with different markers.\n    Can have a numeric dtype but will always be treated as categorical.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.rel.sizes}\n{params.rel.size_order}\n{params.rel.size_norm}\n{params.rel.markers}\n{params.rel.style_order}\n{params.rel.legend}\n{params.core.ax}\nkwargs : key, value mappings\n    Other keyword arguments are passed down to\n    :meth:`matplotlib.axes.Axes.scatter`.\n\nReturns\n-------\n{returns.ax}\n\nSee Also\n--------\n{seealso.lineplot}\n{seealso.stripplot}\n{seealso.swarmplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/scatterplot.rst\n\n'.format(narrative=_relational_narrative, params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
relplot.__doc__ = 'Figure-level interface for drawing relational plots onto a FacetGrid.\n\nThis function provides access to several different axes-level functions\nthat show the relationship between two variables with semantic mappings\nof subsets. The `kind` parameter selects the underlying axes-level\nfunction to use:\n\n- :func:`scatterplot` (with `kind="scatter"`; the default)\n- :func:`lineplot` (with `kind="line"`)\n\nExtra keyword arguments are passed to the underlying function, so you\nshould refer to the documentation for each to see kind-specific options.\n\n{narrative.main_api}\n\n{narrative.relational_semantic}\n\nAfter plotting, the :class:`FacetGrid` with the plot is returned and can\nbe used directly to tweak supporting plot details or add other layers.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\nhue : vector or key in `data`\n    Grouping variable that will produce elements with different colors.\n    Can be either categorical or numeric, although color mapping will\n    behave differently in latter case.\nsize : vector or key in `data`\n    Grouping variable that will produce elements with different sizes.\n    Can be either categorical or numeric, although size mapping will\n    behave differently in latter case.\nstyle : vector or key in `data`\n    Grouping variable that will produce elements with different styles.\n    Can have a numeric dtype but will always be treated as categorical.\n{params.rel.units}\nweights : vector or key in `data`\n    Data values or column used to compute weighted estimation.\n    Note that use of weights currently limits the choice of statistics\n    to a \'mean\' estimator and \'ci\' errorbar.\n{params.facets.rowcol}\n{params.facets.col_wrap}\nrow_order, col_order : lists of strings\n    Order to organize the rows and/or columns of the grid in, otherwise the\n    orders are inferred from the data objects.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n{params.rel.sizes}\n{params.rel.size_order}\n{params.rel.size_norm}\n{params.rel.style_order}\n{params.rel.dashes}\n{params.rel.markers}\n{params.rel.legend}\nkind : string\n    Kind of plot to draw, corresponding to a seaborn relational plot.\n    Options are `"scatter"` or `"line"`.\n{params.facets.height}\n{params.facets.aspect}\nfacet_kws : dict\n    Dictionary of other keyword arguments to pass to :class:`FacetGrid`.\nkwargs : key, value pairings\n    Other keyword arguments are passed through to the underlying plotting\n    function.\n\nReturns\n-------\n{returns.facetgrid}\n\nExamples\n--------\n\n.. include:: ../docstrings/relplot.rst\n\n'.format(narrative=_relational_narrative, params=_param_docs, returns=_core_docs['returns'])
    def _process_data(self):
        """Process the input data and prepare it for plotting."""
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                if 'x' in self.variables and 'y' in self.variables:
                    self.x = self.data[self.variables['x']]
                    self.y = self.data[self.variables['y']]
                    self.x_label = self.variables['x']
                    self.y_label = self.variables['y']
                else:
                    raise ValueError("Both 'x' and 'y' variables must be specified when using a DataFrame.")
            elif isinstance(self.data, dict):
                self.x = self.data.get('x', [])
                self.y = self.data.get('y', [])
                self.x_label = 'x'
                self.y_label = 'y'
            else:
                raise ValueError("Unsupported data type. Please provide a pandas DataFrame or a dictionary.")

            if self.sort:
                sort_idx = np.argsort(self.x)
                self.x = self.x[sort_idx]
                self.y = self.y[sort_idx]

            if 'hue' in self.variables:
                self.colors = self._get_colors(self.data[self.variables['hue']])

    def _get_colors(self, hue_data):
        """Get colors based on the hue variable."""
        n_colors = len(hue_data.unique())
        return color_palette(n_colors=n_colors)
