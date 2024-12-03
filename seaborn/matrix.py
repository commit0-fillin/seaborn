"""Functions to visualize matrices of data."""
import warnings
import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
try:
    from scipy.cluster import hierarchy
    _no_scipy = False
except ImportError:
    _no_scipy = True
from . import cm
from .axisgrid import Grid
from ._compat import get_colormap
from .utils import despine, axis_ticklabels_overlap, relative_luminance, to_utf8, _draw_figure
__all__ = ['heatmap', 'clustermap']

def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return ", ".join(map(str, index.names))
    elif isinstance(index, pd.Index):
        return index.name or ""
    else:
        return ""

def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return [", ".join(map(str, i)) for i in index.values]
    else:
        return index.values

def _convert_colors(colors):
    """Convert either a list of colors or nested lists of colors to RGB."""
    if isinstance(colors, list):
        if isinstance(colors[0], list):
            return [list(map(mpl.colors.to_rgb, sublist)) for sublist in colors]
        else:
            return list(map(mpl.colors.to_rgb, colors))
    else:
        return mpl.colors.to_rgb(colors)

def _matrix_mask(data, mask):
    """Ensure that data and mask are compatible and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, bool)

    if isinstance(mask, np.ndarray):
        mask = pd.DataFrame(mask, index=data.index, columns=data.columns)

    mask = mask.reindex_like(data)

    if np.any(mask.isnull()):
        mask = mask.fillna(True)

    return mask

class _HeatMapper:
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels=True, yticklabels=True, mask=None):
        """Initialize the plotting object."""
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)
        mask = _matrix_mask(data, mask)
        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []
        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []
        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == 'auto':
            self.xticks = 'auto'
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels, xtickevery)
        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == 'auto':
            self.yticks = 'auto'
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels, ytickevery)
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ''
        self.ylabel = ylabel if ylabel is not None else ''
        self._determine_cmap_params(plot_data, vmin, vmax, cmap, center, robust)
        if annot is None or annot is False:
            annot = False
            annot_data = None
        else:
            if isinstance(annot, bool):
                annot_data = plot_data
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != plot_data.shape:
                    err = '`data` and `annot` must have same shape.'
                    raise ValueError(err)
            annot = True
        self.data = data
        self.plot_data = plot_data
        self.annot = annot
        self.annot_data = annot_data
        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws.copy()
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws.copy()

    def _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        pass

    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        pass

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        pass

    def _auto_ticks(self, ax, labels, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        pass

    def plot(self, ax, cax, kws):
        """Draw the heatmap on the provided Axes."""
        pass

def heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs):
    """Plot rectangular data as a color-encoded matrix.

    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergent data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the data. Note that DataFrames will match on position, not index.
    fmt : str, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
        is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : bool, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    See Also
    --------
    clustermap : Plot a matrix using hierarchical clustering to arrange the
                 rows and columns.

    Examples
    --------

    .. include:: ../docstrings/heatmap.rst

    """
    pass

class _DendrogramPlotter:
    """Object for drawing tree of similarities between data rows/columns"""

    def __init__(self, data, linkage, metric, method, axis, label, rotate):
        """Plot a dendrogram of the relationships between the columns of data

        Parameters
        ----------
        data : pandas.DataFrame
            Rectangular data
        """
        self.axis = axis
        if self.axis == 1:
            data = data.T
        if isinstance(data, pd.DataFrame):
            array = data.values
        else:
            array = np.asarray(data)
            data = pd.DataFrame(array)
        self.array = array
        self.data = data
        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.axis = axis
        self.label = label
        self.rotate = rotate
        if linkage is None:
            self.linkage = self.calculated_linkage
        else:
            self.linkage = linkage
        self.dendrogram = self.calculate_dendrogram()
        ticks = 10 * np.arange(self.data.shape[0]) + 5
        if self.label:
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            if self.rotate:
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []
                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            self.xticks, self.yticks = ([], [])
            self.yticklabels, self.xticklabels = ([], [])
            self.xlabel, self.ylabel = ('', '')
        self.dependent_coord = self.dendrogram['dcoord']
        self.independent_coord = self.dendrogram['icoord']

    def calculate_dendrogram(self):
        """Calculates a dendrogram based on the linkage matrix

        Made a separate function, not a property because don't want to
        recalculate the dendrogram every time it is accessed.

        Returns
        -------
        dendrogram : dict
            Dendrogram dictionary as returned by scipy.cluster.hierarchy
            .dendrogram. The important key-value pairing is
            "reordered_ind" which indicates the re-ordering of the matrix
        """
        pass

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        pass

    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted

        """
        pass

def dendrogram(data, *, linkage=None, axis=1, label=True, metric='euclidean', method='average', rotate=False, tree_kws=None, ax=None):
    """Draw a tree diagram of relationships within a matrix

    Parameters
    ----------
    data : pandas.DataFrame
        Rectangular data
    linkage : numpy.array, optional
        Linkage matrix
    axis : int, optional
        Which axis to use to calculate linkage. 0 is rows, 1 is columns.
    label : bool, optional
        If True, label the dendrogram at leaves with column or row names
    metric : str, optional
        Distance metric. Anything valid for scipy.spatial.distance.pdist
    method : str, optional
        Linkage method to use. Anything valid for
        scipy.cluster.hierarchy.linkage
    rotate : bool, optional
        When plotting the matrix, whether to rotate it 90 degrees
        counter-clockwise, so the leaves face right
    tree_kws : dict, optional
        Keyword arguments for the ``matplotlib.collections.LineCollection``
        that is used for plotting the lines of the dendrogram tree.
    ax : matplotlib axis, optional
        Axis to plot on, otherwise uses current axis

    Returns
    -------
    dendrogramplotter : _DendrogramPlotter
        A Dendrogram plotter object.

    Notes
    -----
    Access the reordered dendrogram indices with
    dendrogramplotter.reordered_ind

    """
    pass

class ClusterGrid(Grid):

    def __init__(self, data, pivot_kws=None, z_score=None, standard_scale=None, figsize=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=None, colors_ratio=None, cbar_pos=None):
        """Grid object for organizing clustered heatmap input on to axes"""
        if _no_scipy:
            raise RuntimeError('ClusterGrid requires scipy to be available')
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)
        self.data2d = self.format_data(self.data, pivot_kws, z_score, standard_scale)
        self.mask = _matrix_mask(self.data2d, mask)
        self._figure = plt.figure(figsize=figsize)
        self.row_colors, self.row_color_labels = self._preprocess_colors(data, row_colors, axis=0)
        self.col_colors, self.col_color_labels = self._preprocess_colors(data, col_colors, axis=1)
        try:
            row_dendrogram_ratio, col_dendrogram_ratio = dendrogram_ratio
        except TypeError:
            row_dendrogram_ratio = col_dendrogram_ratio = dendrogram_ratio
        try:
            row_colors_ratio, col_colors_ratio = colors_ratio
        except TypeError:
            row_colors_ratio = col_colors_ratio = colors_ratio
        width_ratios = self.dim_ratios(self.row_colors, row_dendrogram_ratio, row_colors_ratio)
        height_ratios = self.dim_ratios(self.col_colors, col_dendrogram_ratio, col_colors_ratio)
        nrows = 2 if self.col_colors is None else 3
        ncols = 2 if self.row_colors is None else 3
        self.gs = gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
        self.ax_row_dendrogram = self._figure.add_subplot(self.gs[-1, 0])
        self.ax_col_dendrogram = self._figure.add_subplot(self.gs[0, -1])
        self.ax_row_dendrogram.set_axis_off()
        self.ax_col_dendrogram.set_axis_off()
        self.ax_row_colors = None
        self.ax_col_colors = None
        if self.row_colors is not None:
            self.ax_row_colors = self._figure.add_subplot(self.gs[-1, 1])
        if self.col_colors is not None:
            self.ax_col_colors = self._figure.add_subplot(self.gs[1, -1])
        self.ax_heatmap = self._figure.add_subplot(self.gs[-1, -1])
        if cbar_pos is None:
            self.ax_cbar = self.cax = None
        else:
            self.ax_cbar = self._figure.add_subplot(self.gs[0, 0])
            self.cax = self.ax_cbar
        self.cbar_pos = cbar_pos
        self.dendrogram_row = None
        self.dendrogram_col = None

    def _preprocess_colors(self, data, colors, axis):
        """Preprocess {row/col}_colors to extract labels and convert colors."""
        pass

    def format_data(self, data, pivot_kws, z_score=None, standard_scale=None):
        """Extract variables from data or use directly."""
        pass

    @staticmethod
    def z_score(data2d, axis=1):
        """Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.
        """
        pass

    @staticmethod
    def standard_scale(data2d, axis=1):
        """Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        pass

    def dim_ratios(self, colors, dendrogram_ratio, colors_ratio):
        """Get the proportions of the figure taken up by each axes."""
        pass

    @staticmethod
    def color_list_to_matrix_and_cmap(colors, ind, axis=0):
        """Turns a list of colors into a numpy matrix and matplotlib colormap

        These arguments can now be plotted using heatmap(matrix, cmap)
        and the provided colors will be plotted.

        Parameters
        ----------
        colors : list of matplotlib colors
            Colors to label the rows or columns of a dataframe.
        ind : list of ints
            Ordering of the rows or columns, to reorder the original colors
            by the clustered dendrogram order
        axis : int
            Which axis this is labeling

        Returns
        -------
        matrix : numpy.array
            A numpy array of integer values, where each indexes into the cmap
        cmap : matplotlib.colors.ListedColormap

        """
        pass

    def plot_colors(self, xind, yind, **kws):
        """Plots color labels between the dendrogram and the heatmap

        Parameters
        ----------
        heatmap_kws : dict
            Keyword arguments heatmap

        """
        pass

def clustermap(data, *, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None, figsize=(10, 10), cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=0.2, colors_ratio=0.03, cbar_pos=(0.02, 0.8, 0.05, 0.18), tree_kws=None, **kwargs):
    """
    Plot a matrix dataset as a hierarchically-clustered heatmap.

    This function requires scipy to be available.

    Parameters
    ----------
    data : 2D array-like
        Rectangular data for clustering. Cannot contain NAs.
    pivot_kws : dict, optional
        If `data` is a tidy dataframe, can provide keyword arguments for
        pivot to create a rectangular dataframe.
    method : str, optional
        Linkage method to use for calculating clusters. See
        :func:`scipy.cluster.hierarchy.linkage` documentation for more
        information.
    metric : str, optional
        Distance metric to use for the data. See
        :func:`scipy.spatial.distance.pdist` documentation for more options.
        To use different metrics (or methods) for rows and columns, you may
        construct each linkage matrix yourself and provide them as
        `{row,col}_linkage`.
    z_score : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to calculate z-scores
        for the rows or the columns. Z scores are: z = (x - mean)/std, so
        values in each row (column) will get the mean of the row (column)
        subtracted, then divided by the standard deviation of the row (column).
        This ensures that each row (column) has mean of 0 and variance of 1.
    standard_scale : int or None, optional
        Either 0 (rows) or 1 (columns). Whether or not to standardize that
        dimension, meaning for each row or column, subtract the minimum and
        divide each by its maximum.
    figsize : tuple of (width, height), optional
        Overall size of the figure.
    cbar_kws : dict, optional
        Keyword arguments to pass to `cbar_kws` in :func:`heatmap`, e.g. to
        add a label to the colorbar.
    {row,col}_cluster : bool, optional
        If ``True``, cluster the {rows, columns}.
    {row,col}_linkage : :class:`numpy.ndarray`, optional
        Precomputed linkage matrix for the rows or columns. See
        :func:`scipy.cluster.hierarchy.linkage` for specific formats.
    {row,col}_colors : list-like or pandas DataFrame/Series, optional
        List of colors to label for either the rows or columns. Useful to evaluate
        whether samples within a group are clustered together. Can use nested lists or
        DataFrame for multiple color levels of labeling. If given as a
        :class:`pandas.DataFrame` or :class:`pandas.Series`, labels for the colors are
        extracted from the DataFrames column names or from the name of the Series.
        DataFrame/Series colors are also matched to the data by their index, ensuring
        colors are drawn in the correct order.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where `mask` is True.
        Cells with missing values are automatically masked. Only used for
        visualizing, not for calculating.
    {dendrogram,colors}_ratio : float, or pair of floats, optional
        Proportion of the figure size devoted to the two marginal elements. If
        a pair is given, they correspond to (row, col) ratios.
    cbar_pos : tuple of (left, bottom, width, height), optional
        Position of the colorbar axes in the figure. Setting to ``None`` will
        disable the colorbar.
    tree_kws : dict, optional
        Parameters for the :class:`matplotlib.collections.LineCollection`
        that is used to plot the lines of the dendrogram tree.
    kwargs : other keyword arguments
        All other keyword arguments are passed to :func:`heatmap`.

    Returns
    -------
    :class:`ClusterGrid`
        A :class:`ClusterGrid` instance.

    See Also
    --------
    heatmap : Plot rectangular data as a color-encoded matrix.

    Notes
    -----
    The returned object has a ``savefig`` method that should be used if you
    want to save the figure object without clipping the dendrograms.

    To access the reordered row indices, use:
    ``clustergrid.dendrogram_row.reordered_ind``

    Column indices, use:
    ``clustergrid.dendrogram_col.reordered_ind``

    Examples
    --------

    .. include:: ../docstrings/clustermap.rst

    """
    pass
