from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import adjust_legend_subtitles, set_hls_values, _check_argument, _draw_figure, _disable_autolayout
from .palettes import color_palette, blend_palette
from ._docstrings import DocstringComponents, _core_docs
__all__ = ['FacetGrid', 'PairGrid', 'JointGrid', 'pairplot', 'jointplot']
_param_docs = DocstringComponents.from_nested_components(core=_core_docs['params'])

class _BaseGrid:
    """Base class for grids of subplots."""

    def set(self, **kwargs):
        """Set attributes on each subplot Axes."""
        for ax in self.axes.flat:
            if ax is not None:
                ax.set(**kwargs)
        return self

    @property
    def fig(self):
        """DEPRECATED: prefer the `figure` property."""
        import warnings
        warnings.warn("The `fig` property is deprecated. Use `figure` instead.", DeprecationWarning)
        return self.figure

    @property
    def figure(self):
        """Access the :class:`matplotlib.figure.Figure` object underlying the grid."""
        return self._figure

    def apply(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return self.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` is ignored; this method returns self.
        See the `pipe` method if you want the return value.

        Added in v0.12.0.

        """
        func(self, *args, **kwargs)
        return self

    def pipe(self, func, *args, **kwargs):
        """
        Pass the grid to a user-supplied function and return its value.

        The `func` must accept an object of this type for its first
        positional argument. Additional arguments are passed through.
        The return value of `func` becomes the return value of this method.
        See the `apply` method if you want to return self instead.

        Added in v0.12.0.

        """
        return func(self, *args, **kwargs)

    def savefig(self, *args, **kwargs):
        """
        Save an image of the plot.

        This wraps :meth:`matplotlib.figure.Figure.savefig`, using bbox_inches="tight"
        by default. Parameters are passed through to the matplotlib function.

        """
        kwargs.setdefault("bbox_inches", "tight")
        self.figure.savefig(*args, **kwargs)

class Grid(_BaseGrid):
    """A grid that can have multiple subplots and an external legend."""
    _margin_titles = False
    _legend_out = True

    def __init__(self):
        self._tight_layout_rect = [0, 0, 1, 1]
        self._tight_layout_pad = None
        self._extract_legend_handles = False

    def tight_layout(self, *args, **kwargs):
        """Call fig.tight_layout within rect that exclude the legend."""
        pass

    def add_legend(self, legend_data=None, title=None, label_order=None, adjust_subtitles=False, **kwargs):
        """Draw a legend, maybe placing it outside axes and resizing the figure.

        Parameters
        ----------
        legend_data : dict
            Dictionary mapping label names (or two-element tuples where the
            second element is a label name) to matplotlib artist handles. The
            default reads from ``self._legend_data``.
        title : string
            Title for the legend. The default reads from ``self._hue_var``.
        label_order : list of labels
            The order that the legend entries should appear in. The default
            reads from ``self.hue_names``.
        adjust_subtitles : bool
            If True, modify entries with invisible artists to left-align
            the labels and set the font size to that of a title.
        kwargs : key, value pairings
            Other keyword arguments are passed to the underlying legend methods
            on the Figure or Axes object.

        Returns
        -------
        self : Grid instance
            Returns self for easy chaining.

        """
        pass

    def _update_legend_data(self, ax):
        """Extract the legend data from an axes object and save it."""
        pass

    def _get_palette(self, data, hue, hue_order, palette):
        """Get a list of colors for the hue variable."""
        pass

    @property
    def legend(self):
        """The :class:`matplotlib.legend.Legend` object, if present."""
        pass

    def tick_params(self, axis='both', **kwargs):
        """Modify the ticks, tick labels, and gridlines.

        Parameters
        ----------
        axis : {'x', 'y', 'both'}
            The axis on which to apply the formatting.
        kwargs : keyword arguments
            Additional keyword arguments to pass to
            :meth:`matplotlib.axes.Axes.tick_params`.

        Returns
        -------
        self : Grid instance
            Returns self for easy chaining.

        """
        pass
_facet_docs = dict(data=dedent('    data : DataFrame\n        Tidy ("long-form") dataframe where each column is a variable and each\n        row is an observation.    '), rowcol=dedent('    row, col : vectors or keys in ``data``\n        Variables that define subsets to plot on different facets.    '), rowcol_order=dedent('    {row,col}_order : vector of strings\n        Specify the order in which levels of the ``row`` and/or ``col`` variables\n        appear in the grid of subplots.    '), col_wrap=dedent('    col_wrap : int\n        "Wrap" the column variable at this width, so that the column facets\n        span multiple rows. Incompatible with a ``row`` facet.    '), share_xy=dedent("    share{x,y} : bool, 'col', or 'row' optional\n        If true, the facets will share y axes across columns and/or x axes\n        across rows.    "), height=dedent('    height : scalar\n        Height (in inches) of each facet. See also: ``aspect``.    '), aspect=dedent('    aspect : scalar\n        Aspect ratio of each facet, so that ``aspect * height`` gives the width\n        of each facet in inches.    '), palette=dedent('    palette : palette name, list, or dict\n        Colors to use for the different levels of the ``hue`` variable. Should\n        be something that can be interpreted by :func:`color_palette`, or a\n        dictionary mapping hue levels to matplotlib colors.    '), legend_out=dedent('    legend_out : bool\n        If ``True``, the figure size will be extended, and the legend will be\n        drawn outside the plot on the center right.    '), margin_titles=dedent('    margin_titles : bool\n        If ``True``, the titles for the row variable are drawn to the right of\n        the last column. This option is experimental and may not work in all\n        cases.    '), facet_kws=dedent('    facet_kws : dict\n        Additional parameters passed to :class:`FacetGrid`.\n    '))

class FacetGrid(Grid):
    """Multi-plot grid for plotting conditional relationships."""

    def __init__(self, data, *, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=False, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None):
        super().__init__()
        data = handle_data_source(data)
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = categorical_order(data[hue], hue_order)
        colors = self._get_palette(data, hue, hue_order, palette)
        if row is None:
            row_names = []
        else:
            row_names = categorical_order(data[row], row_order)
        if col is None:
            col_names = []
        else:
            col_names = categorical_order(data[col], col_order)
        hue_kws = hue_kws if hue_kws is not None else {}
        none_na = np.zeros(len(data), bool)
        if dropna:
            row_na = none_na if row is None else data[row].isnull()
            col_na = none_na if col is None else data[col].isnull()
            hue_na = none_na if hue is None else data[hue].isnull()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow
        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = 'Cannot use `row` and `col_wrap` together.'
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow
        figsize = (ncol * height * aspect, nrow * height)
        if col_wrap is not None:
            margin_titles = False
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws['xlim'] = xlim
        if ylim is not None:
            subplot_kws['ylim'] = ylim
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        if col_wrap is None:
            kwargs = dict(squeeze=False, sharex=sharex, sharey=sharey, subplot_kw=subplot_kws, gridspec_kw=gridspec_kws)
            axes = fig.subplots(nrow, ncol, **kwargs)
            if col is None and row is None:
                axes_dict = {}
            elif col is None:
                axes_dict = dict(zip(row_names, axes.flat))
            elif row is None:
                axes_dict = dict(zip(col_names, axes.flat))
            else:
                facet_product = product(row_names, col_names)
                axes_dict = dict(zip(facet_product, axes.flat))
        else:
            if gridspec_kws:
                warnings.warn('`gridspec_kws` ignored when using `col_wrap`')
            n_axes = len(col_names)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws['sharex'] = axes[0]
            if sharey:
                subplot_kws['sharey'] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)
            axes_dict = dict(zip(col_names, axes))
        self._figure = fig
        self._axes = axes
        self._axes_dict = axes_dict
        self._legend = None
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._margin_titles = margin_titles
        self._margin_titles_texts = []
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._sharex = sharex
        self._sharey = sharey
        self._dropna = dropna
        self._not_na = not_na
        self.set_titles()
        self.tight_layout()
        if despine:
            self.despine()
        if sharex in [True, 'col']:
            for ax in self._not_bottom_axes:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
                ax.xaxis.label.set_visible(False)
        if sharey in [True, 'row']:
            for ax in self._not_left_axes:
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
                ax.yaxis.label.set_visible(False)
    __init__.__doc__ = dedent('        Initialize the matplotlib figure and FacetGrid object.\n\n        This class maps a dataset onto multiple axes arrayed in a grid of rows\n        and columns that correspond to *levels* of variables in the dataset.\n        The plots it produces are often called "lattice", "trellis", or\n        "small-multiple" graphics.\n\n        It can also represent levels of a third variable with the ``hue``\n        parameter, which plots different subsets of data in different colors.\n        This uses color to resolve elements on a third dimension, but only\n        draws subsets on top of each other and will not tailor the ``hue``\n        parameter for the specific visualization the way that axes-level\n        functions that accept ``hue`` will.\n\n        The basic workflow is to initialize the :class:`FacetGrid` object with\n        the dataset and the variables that are used to structure the grid. Then\n        one or more plotting functions can be applied to each subset by calling\n        :meth:`FacetGrid.map` or :meth:`FacetGrid.map_dataframe`. Finally, the\n        plot can be tweaked with other methods to do things like change the\n        axis labels, use different ticks, or add a legend. See the detailed\n        code examples below for more information.\n\n        .. warning::\n\n            When using seaborn functions that infer semantic mappings from a\n            dataset, care must be taken to synchronize those mappings across\n            facets (e.g., by defining the ``hue`` mapping with a palette dict or\n            setting the data type of the variables to ``category``). In most cases,\n            it will be better to use a figure-level function (e.g. :func:`relplot`\n            or :func:`catplot`) than to use :class:`FacetGrid` directly.\n\n        See the :ref:`tutorial <grid_tutorial>` for more information.\n\n        Parameters\n        ----------\n        {data}\n        row, col, hue : strings\n            Variables that define subsets of the data, which will be drawn on\n            separate facets in the grid. See the ``{{var}}_order`` parameters to\n            control the order of levels of this variable.\n        {col_wrap}\n        {share_xy}\n        {height}\n        {aspect}\n        {palette}\n        {{row,col,hue}}_order : lists\n            Order for the levels of the faceting variables. By default, this\n            will be the order that the levels appear in ``data`` or, if the\n            variables are pandas categoricals, the category order.\n        hue_kws : dictionary of param -> list of values mapping\n            Other keyword arguments to insert into the plotting call to let\n            other plot attributes vary across levels of the hue variable (e.g.\n            the markers in a scatterplot).\n        {legend_out}\n        despine : boolean\n            Remove the top and right spines from the plots.\n        {margin_titles}\n        {{x, y}}lim: tuples\n            Limits for each of the axes on each facet (only relevant when\n            share{{x, y}} is True).\n        subplot_kws : dict\n            Dictionary of keyword arguments passed to matplotlib subplot(s)\n            methods.\n        gridspec_kws : dict\n            Dictionary of keyword arguments passed to\n            :class:`matplotlib.gridspec.GridSpec`\n            (via :meth:`matplotlib.figure.Figure.subplots`).\n            Ignored if ``col_wrap`` is not ``None``.\n\n        See Also\n        --------\n        PairGrid : Subplot grid for plotting pairwise relationships\n        relplot : Combine a relational plot and a :class:`FacetGrid`\n        displot : Combine a distribution plot and a :class:`FacetGrid`\n        catplot : Combine a categorical plot and a :class:`FacetGrid`\n        lmplot : Combine a regression plot and a :class:`FacetGrid`\n\n        Examples\n        --------\n\n        .. note::\n\n            These examples use seaborn functions to demonstrate some of the\n            advanced features of the class, but in most cases you will want\n            to use figue-level functions (e.g. :func:`displot`, :func:`relplot`)\n            to make the plots shown here.\n\n        .. include:: ../docstrings/FacetGrid.rst\n\n        ').format(**_facet_docs)

    def facet_data(self):
        """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
        pass

    def map(self, func, *args, **kwargs):
        """Apply a plotting function to each facet's subset of the data.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. It
            must plot to the currently active matplotlib Axes and take a
            `color` keyword argument. If faceting on the `hue` dimension,
            it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """
        pass

    def map_dataframe(self, func, *args, **kwargs):
        """Like ``.map`` but passes args as strings and inserts data in kwargs.

        This method is suitable for plotting with functions that accept a
        long-form DataFrame as a `data` keyword argument and access the
        data in that DataFrame using string variable names.

        Parameters
        ----------
        func : callable
            A plotting function that takes data and keyword arguments. Unlike
            the `map` method, a function used here must "understand" Pandas
            objects. It also must plot to the currently active matplotlib Axes
            and take a `color` keyword argument. If faceting on the `hue`
            dimension, it must also take a `label` keyword argument.
        args : strings
            Column names in self.data that identify variables with data to
            plot. The data for each variable is passed to `func` in the
            order the variables are specified in the call.
        kwargs : keyword arguments
            All keyword arguments are passed to the plotting function.

        Returns
        -------
        self : object
            Returns self.

        """
        pass

    def _finalize_grid(self, axlabels):
        """Finalize the annotations and layout."""
        pass

    def facet_axis(self, row_i, col_j, modify_state=True):
        """Make the axis identified by these indices active and return it."""
        pass

    def despine(self, **kwargs):
        """Remove axis spines from the facets."""
        pass

    def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
        """Set axis labels on the left column and bottom row of the grid."""
        pass

    def set_xlabels(self, label=None, clear_inner=True, **kwargs):
        """Label the x axis on the bottom row of the grid."""
        pass

    def set_ylabels(self, label=None, clear_inner=True, **kwargs):
        """Label the y axis on the left column of the grid."""
        pass

    def set_xticklabels(self, labels=None, step=None, **kwargs):
        """Set x axis tick labels of the grid."""
        pass

    def set_yticklabels(self, labels=None, **kwargs):
        """Set y axis tick labels on the left column of the grid."""
        pass

    def set_titles(self, template=None, row_template=None, col_template=None, **kwargs):
        """Draw titles either above each facet or on the grid margins.

        Parameters
        ----------
        template : string
            Template for all titles with the formatting keys {col_var} and
            {col_name} (if using a `col` faceting variable) and/or {row_var}
            and {row_name} (if using a `row` faceting variable).
        row_template:
            Template for the row variable when titles are drawn on the grid
            margins. Must have {row_var} and {row_name} formatting keys.
        col_template:
            Template for the column variable when titles are drawn on the grid
            margins. Must have {col_var} and {col_name} formatting keys.

        Returns
        -------
        self: object
            Returns self.

        """
        pass

    def refline(self, *, x=None, y=None, color='.5', linestyle='--', **line_kws):
        """Add a reference line(s) to each facet.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s). Pass ``color=None`` to
            use ``hue`` mapping.
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`FacetGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass

    @property
    def axes(self):
        """An array of the :class:`matplotlib.axes.Axes` objects in the grid."""
        pass

    @property
    def ax(self):
        """The :class:`matplotlib.axes.Axes` when no faceting variables are assigned."""
        pass

    @property
    def axes_dict(self):
        """A mapping of facet names to corresponding :class:`matplotlib.axes.Axes`.

        If only one of ``row`` or ``col`` is assigned, each key is a string
        representing a level of that variable. If both facet dimensions are
        assigned, each key is a ``({row_level}, {col_level})`` tuple.

        """
        pass

    @property
    def _inner_axes(self):
        """Return a flat array of the inner axes."""
        pass

    @property
    def _left_axes(self):
        """Return a flat array of the left column of axes."""
        pass

    @property
    def _not_left_axes(self):
        """Return a flat array of axes that aren't on the left column."""
        pass

    @property
    def _bottom_axes(self):
        """Return a flat array of the bottom row of axes."""
        pass

    @property
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        pass

class PairGrid(Grid):
    """Subplot grid for plotting pairwise relationships in a dataset.

    This object maps each variable in a dataset onto a column and row in a
    grid of multiple axes. Different axes-level plotting functions can be
    used to draw bivariate plots in the upper and lower triangles, and the
    marginal distribution of each variable can be shown on the diagonal.

    Several different common plots can be generated in a single line using
    :func:`pairplot`. Use :class:`PairGrid` when you need more flexibility.

    See the :ref:`tutorial <grid_tutorial>` for more information.

    """

    def __init__(self, data, *, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None, corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False):
        """Initialize the plot figure and PairGrid object.

        Parameters
        ----------
        data : DataFrame
            Tidy (long-form) dataframe where each column is a variable and
            each row is an observation.
        hue : string (variable name)
            Variable in ``data`` to map plot aspects to different colors. This
            variable will be excluded from the default x and y variables.
        vars : list of variable names
            Variables within ``data`` to use, otherwise use every column with
            a numeric datatype.
        {x, y}_vars : lists of variable names
            Variables within ``data`` to use separately for the rows and
            columns of the figure; i.e. to make a non-square plot.
        hue_order : list of strings
            Order for the levels of the hue variable in the palette
        palette : dict or seaborn color palette
            Set of colors for mapping the ``hue`` variable. If a dict, keys
            should be values  in the ``hue`` variable.
        hue_kws : dictionary of param -> list of values mapping
            Other keyword arguments to insert into the plotting call to let
            other plot attributes vary across levels of the hue variable (e.g.
            the markers in a scatterplot).
        corner : bool
            If True, don't add axes to the upper (off-diagonal) triangle of the
            grid, making this a "corner" plot.
        height : scalar
            Height (in inches) of each facet.
        aspect : scalar
            Aspect * height gives the width (in inches) of each facet.
        layout_pad : scalar
            Padding between axes; passed to ``fig.tight_layout``.
        despine : boolean
            Remove the top and right spines from the plots.
        dropna : boolean
            Drop missing values from the data before plotting.

        See Also
        --------
        pairplot : Easily drawing common uses of :class:`PairGrid`.
        FacetGrid : Subplot grid for plotting conditional relationships.

        Examples
        --------

        .. include:: ../docstrings/PairGrid.rst

        """
        super().__init__()
        data = handle_data_source(data)
        numeric_cols = self._find_numeric_cols(data)
        if hue in numeric_cols:
            numeric_cols.remove(hue)
        if vars is not None:
            x_vars = list(vars)
            y_vars = list(vars)
        if x_vars is None:
            x_vars = numeric_cols
        if y_vars is None:
            y_vars = numeric_cols
        if np.isscalar(x_vars):
            x_vars = [x_vars]
        if np.isscalar(y_vars):
            y_vars = [y_vars]
        self.x_vars = x_vars = list(x_vars)
        self.y_vars = y_vars = list(y_vars)
        self.square_grid = self.x_vars == self.y_vars
        if not x_vars:
            raise ValueError('No variables found for grid columns.')
        if not y_vars:
            raise ValueError('No variables found for grid rows.')
        figsize = (len(x_vars) * height * aspect, len(y_vars) * height)
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        axes = fig.subplots(len(y_vars), len(x_vars), sharex='col', sharey='row', squeeze=False)
        self._corner = corner
        if corner:
            hide_indices = np.triu_indices_from(axes, 1)
            for i, j in zip(*hide_indices):
                axes[i, j].remove()
                axes[i, j] = None
        self._figure = fig
        self.axes = axes
        self.data = data
        self.diag_sharey = diag_sharey
        self.diag_vars = None
        self.diag_axes = None
        self._dropna = dropna
        self._add_axis_labels()
        self._hue_var = hue
        if hue is None:
            self.hue_names = hue_order = ['_nolegend_']
            self.hue_vals = pd.Series(['_nolegend_'] * len(data), index=data.index)
        else:
            hue_names = hue_order = categorical_order(data[hue], hue_order)
            if dropna:
                hue_names = list(filter(pd.notnull, hue_names))
            self.hue_names = hue_names
            self.hue_vals = data[hue]
        self.hue_kws = hue_kws if hue_kws is not None else {}
        self._orig_palette = palette
        self._hue_order = hue_order
        self.palette = self._get_palette(data, hue, hue_order, palette)
        self._legend_data = {}
        for ax in axes[:-1, :].flat:
            if ax is None:
                continue
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.label.set_visible(False)
        for ax in axes[:, 1:].flat:
            if ax is None:
                continue
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.label.set_visible(False)
        self._tight_layout_rect = [0.01, 0.01, 0.99, 0.99]
        self._tight_layout_pad = layout_pad
        self._despine = despine
        if despine:
            utils.despine(fig=fig)
        self.tight_layout(pad=layout_pad)

    def map(self, func, **kwargs):
        """Plot with the same function in every subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        pass

    def map_lower(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        pass

    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        pass

    def map_offdiag(self, func, **kwargs):
        """Plot with a bivariate function on the off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        pass

    def map_diag(self, func, **kwargs):
        """Plot with a univariate function on each diagonal subplot.

        Parameters
        ----------
        func : callable plotting function
            Must take an x array as a positional argument and draw onto the
            "currently active" matplotlib Axes. Also needs to accept kwargs
            called ``color`` and  ``label``.

        """
        pass

    def _map_diag_iter_hue(self, func, **kwargs):
        """Put marginal plot on each diagonal axes, iterating over hue."""
        pass

    def _map_bivariate(self, func, indices, **kwargs):
        """Draw a bivariate plot on the indicated axes."""
        pass

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot on the specified axes."""
        pass

    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        """Draw a bivariate plot while iterating over hue subsets."""
        pass

    def _add_axis_labels(self):
        """Add labels to the left and bottom Axes."""
        pass

    def _find_numeric_cols(self, data):
        """Find which variables in a DataFrame are numeric."""
        pass

class JointGrid(_BaseGrid):
    """Grid for drawing a bivariate plot with marginal univariate plots.

    Many plots can be drawn by using the figure-level interface :func:`jointplot`.
    Use this class directly when you need more flexibility.

    """

    def __init__(self, data=None, *, x=None, y=None, hue=None, height=6, ratio=5, space=0.2, palette=None, hue_order=None, hue_norm=None, dropna=False, xlim=None, ylim=None, marginal_ticks=False):
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)
        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
        self._figure = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)
        if not marginal_ticks:
            plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)
            ax_marg_x.yaxis.grid(False)
            ax_marg_y.xaxis.grid(False)
        p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
        plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]
        if dropna:
            plot_data = plot_data.dropna()

        def get_var(var):
            vector = plot_data.get(var, None)
            if vector is not None:
                vector = vector.rename(p.variables.get(var, None))
            return vector
        self.x = get_var('x')
        self.y = get_var('y')
        self.hue = get_var('hue')
        for axis in 'xy':
            name = p.variables.get(axis, None)
            if name is not None:
                getattr(ax_joint, f'set_{axis}label')(name)
        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)
        self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)
        utils.despine(f)
        if not marginal_ticks:
            utils.despine(ax=ax_marg_x, left=True)
            utils.despine(ax=ax_marg_y, bottom=True)
        for axes in [ax_marg_x, ax_marg_y]:
            for axis in [axes.xaxis, axes.yaxis]:
                axis.label.set_visible(False)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def _inject_kwargs(self, func, kws, params):
        """Add params to kws if they are accepted by func."""
        pass

    def plot(self, joint_func, marginal_func, **kwargs):
        """Draw the plot by passing functions for joint and marginal axes.

        This method passes the ``kwargs`` dictionary to both functions. If you
        need more control, call :meth:`JointGrid.plot_joint` and
        :meth:`JointGrid.plot_marginals` directly with specific parameters.

        Parameters
        ----------
        joint_func, marginal_func : callables
            Functions to draw the bivariate and univariate plots. See methods
            referenced above for information about the required characteristics
            of these functions.
        kwargs
            Additional keyword arguments are passed to both functions.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass

    def plot_joint(self, func, **kwargs):
        """Draw a bivariate plot on the joint axes of the grid.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should accept ``x`` and ``y``. Otherwise,
            it must accept ``x`` and ``y`` vectors of data as the first two
            positional arguments, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, the function must
            accept ``hue`` as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass

    def plot_marginals(self, func, **kwargs):
        """Draw univariate plots on each marginal axes.

        Parameters
        ----------
        func : plotting callable
            If a seaborn function, it should  accept ``x`` and ``y`` and plot
            when only one of them is defined. Otherwise, it must accept a vector
            of data as the first positional argument and determine its orientation
            using the ``vertical`` parameter, and it must plot on the "current" axes.
            If ``hue`` was defined in the class constructor, it must accept ``hue``
            as a parameter.
        kwargs
            Keyword argument are passed to the plotting function.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass

    def refline(self, *, x=None, y=None, joint=True, marginal=True, color='.5', linestyle='--', **line_kws):
        """Add a reference line(s) to joint and/or marginal axes.

        Parameters
        ----------
        x, y : numeric
            Value(s) to draw the line(s) at.
        joint, marginal : bools
            Whether to add the reference line(s) to the joint/marginal axes.
        color : :mod:`matplotlib color <matplotlib.colors>`
            Specifies the color of the reference line(s).
        linestyle : str
            Specifies the style of the reference line(s).
        line_kws : key, value mappings
            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`
            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``
            is not None.

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass

    def set_axis_labels(self, xlabel='', ylabel='', **kwargs):
        """Set axis labels on the bivariate axes.

        Parameters
        ----------
        xlabel, ylabel : strings
            Label names for the x and y variables.
        kwargs : key, value mappings
            Other keyword arguments are passed to the following functions:

            - :meth:`matplotlib.axes.Axes.set_xlabel`
            - :meth:`matplotlib.axes.Axes.set_ylabel`

        Returns
        -------
        :class:`JointGrid` instance
            Returns ``self`` for easy method chaining.

        """
        pass
JointGrid.__init__.__doc__ = 'Set up the grid of subplots and store data internally for easy plotting.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\nheight : number\n    Size of each side of the figure in inches (it will be square).\nratio : number\n    Ratio of joint axes height to marginal axes height.\nspace : number\n    Space between the joint and marginal axes\ndropna : bool\n    If True, remove missing observations before plotting.\n{{x, y}}lim : pairs of numbers\n    Set axis limits to these values before plotting.\nmarginal_ticks : bool\n    If False, suppress ticks on the count/density axis of the marginal plots.\n{params.core.hue}\n    Note: unlike in :class:`FacetGrid` or :class:`PairGrid`, the axes-level\n    functions must support ``hue`` to use it in :class:`JointGrid`.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n\nSee Also\n--------\n{seealso.jointplot}\n{seealso.pairgrid}\n{seealso.pairplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/JointGrid.rst\n\n'.format(params=_param_docs, seealso=_core_docs['seealso'])

def pairplot(data, *, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None):
    """Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each numeric
    variable in ``data`` will by shared across the y-axes across a single row and
    the x-axes across a single column. The diagonal plots are treated
    differently: a univariate distribution plot is drawn to show the marginal
    distribution of the data in each column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns.

    This is a high-level interface for :class:`PairGrid` that is intended to
    make it easy to draw a few common styles. You should use :class:`PairGrid`
    directly if you need more flexibility.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : name of variable in ``data``
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    vars : list of variable names
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    kind : {'scatter', 'kde', 'hist', 'reg'}
        Kind of plot to make.
    diag_kind : {'auto', 'hist', 'kde', None}
        Kind of plot for the diagonal subplots. If 'auto', choose based on
        whether or not ``hue`` is used.
    markers : single matplotlib marker code or list
        Either the marker to use for all scatterplot points or a list of markers
        with a length the same as the number of levels in the hue variable so that
        differently colored points will also have different scatterplot
        markers.
    height : scalar
        Height (in inches) of each facet.
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    dropna : boolean
        Drop missing values from the data before plotting.
    {plot, diag, grid}_kws : dicts
        Dictionaries of keyword arguments. ``plot_kws`` are passed to the
        bivariate plotting function, ``diag_kws`` are passed to the univariate
        plotting function, and ``grid_kws`` are passed to the :class:`PairGrid`
        constructor.

    Returns
    -------
    grid : :class:`PairGrid`
        Returns the underlying :class:`PairGrid` instance for further tweaking.

    See Also
    --------
    PairGrid : Subplot grid for more flexible plotting of pairwise relationships.
    JointGrid : Grid for plotting joint and marginal distributions of two variables.

    Examples
    --------

    .. include:: ../docstrings/pairplot.rst

    """
    pass
jointplot.__doc__ = 'Draw a plot of two variables with bivariate and univariate graphs.\n\nThis function provides a convenient interface to the :class:`JointGrid`\nclass, with several canned plot kinds. This is intended to be a fairly\nlightweight wrapper; if you need more flexibility, you should use\n:class:`JointGrid` directly.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nkind : {{ "scatter" | "kde" | "hist" | "hex" | "reg" | "resid" }}\n    Kind of plot to draw. See the examples for references to the underlying functions.\nheight : numeric\n    Size of the figure (it will be square).\nratio : numeric\n    Ratio of joint axes height to marginal axes height.\nspace : numeric\n    Space between the joint and marginal axes\ndropna : bool\n    If True, remove observations that are missing from ``x`` and ``y``.\n{{x, y}}lim : pairs of numbers\n    Axis limits to set before plotting.\n{params.core.color}\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\nmarginal_ticks : bool\n    If False, suppress ticks on the count/density axis of the marginal plots.\n{{joint, marginal}}_kws : dicts\n    Additional keyword arguments for the plot components.\nkwargs\n    Additional keyword arguments are passed to the function used to\n    draw the plot on the joint Axes, superseding items in the\n    ``joint_kws`` dictionary.\n\nReturns\n-------\n{returns.jointgrid}\n\nSee Also\n--------\n{seealso.jointgrid}\n{seealso.pairgrid}\n{seealso.pairplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/jointplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])
