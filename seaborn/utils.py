"""Utility functions, mostly for internal use."""
import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
__all__ = ['desaturate', 'saturate', 'set_hls_values', 'move_legend', 'despine', 'get_dataset_names', 'get_data_home', 'load_dataset']
DATASET_SOURCE = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master'
DATASET_NAMES_URL = f'{DATASET_SOURCE}/dataset_names.txt'

def ci_to_errsize(cis, heights):
    """Convert intervals to error arguments relative to plot heights.

    Parameters
    ----------
    cis : 2 x n sequence
        sequence of confidence interval limits
    heights : n sequence
        sequence of plot heights

    Returns
    -------
    errsize : 2 x n array
        sequence of error size relative to height values in correct
        format as argument for plt.bar

    """
    cis = np.asarray(cis)
    heights = np.asarray(heights)
    errsize = []
    for i, (low, high) in enumerate(np.transpose(cis)):
        h = heights[i]
        errsize.append([h - low, high - h])
    return np.transpose(errsize)

def _draw_figure(fig):
    """Force draw of a matplotlib figure, accounting for back-compat."""
    if hasattr(fig.canvas, "draw_idle"):
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()

def _default_color(method, hue, color, kws, saturation=1):
    """If needed, get a default color by using the matplotlib property cycle."""
    if color is not None:
        return color
    elif hue is not None:
        return None
    elif method.__name__.startswith("scatter"):
        return plt.rcParams["patch.facecolor"]
    else:
        color = next(plt.gca()._get_lines.prop_cycler)["color"]
        if saturation < 1:
            color = desaturate(color, saturation)
        return color

def desaturate(color, prop):
    """Decrease the saturation channel of a color by some percent.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    prop : float
        saturation channel of color will be multiplied by this value

    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation

    """
    import matplotlib.colors as mplcol
    rgb = mplcol.to_rgb(color)
    hsv = mplcol.rgb_to_hsv(rgb)
    hsv[1] = hsv[1] * prop
    return mplcol.hsv_to_rgb(hsv)

def saturate(color):
    """Return a fully saturated color with the same hue.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name

    Returns
    -------
    new_color : rgb tuple
        saturated color code in RGB tuple representation

    """
    import matplotlib.colors as mplcol
    rgb = mplcol.to_rgb(color)
    hsv = mplcol.rgb_to_hsv(rgb)
    hsv[1] = 1.0
    return mplcol.hsv_to_rgb(hsv)

def set_hls_values(color, h=None, l=None, s=None):
    """Independently manipulate the h, l, or s channels of a color.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    h, l, s : floats between 0 and 1, or None
        new values for each channel in hls space

    Returns
    -------
    new_color : rgb tuple
        new color code in RGB tuple representation

    """
    import matplotlib.colors as mplcol
    rgb = mplcol.to_rgb(color)
    hls = list(mplcol.rgb_to_hls(*rgb))
    if h is not None:
        hls[0] = h
    if l is not None:
        hls[1] = l
    if s is not None:
        hls[2] = s
    return mplcol.hls_to_rgb(*hls)

def axlabel(xlabel, ylabel, **kwargs):
    """Grab current axis and label it.

    DEPRECATED: will be removed in a future version.

    """
    import warnings
    warnings.warn("axlabel is deprecated and will be removed in a future version. "
                  "Use ax.set(xlabel=..., ylabel=...) instead.",
                  FutureWarning)
    ax = plt.gca()
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)

def remove_na(vector):
    """Helper method for removing null values from data vectors.

    Parameters
    ----------
    vector : vector object
        Must implement boolean masking with [] subscript syntax.

    Returns
    -------
    clean_clean : same type as ``vector``
        Vector of data with null values removed. May be a copy or a view.

    """
    return vector[pd.notnull(vector)]

def get_color_cycle():
    """Return the list of colors in the current matplotlib color cycle

    Parameters
    ----------
    None

    Returns
    -------
    colors : list
        List of matplotlib colors in the current cycle, or dark gray if
        the current color cycle is empty.
    """
    from matplotlib import rcParams
    colors = rcParams.get('axes.prop_cycle', rcParams.get('axes.color_cycle', ['#333333']))
    if isinstance(colors, cycler):
        colors = [c['color'] for c in colors]
    return colors

def despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.

    Returns
    -------
    None

    """
    pass

def move_legend(obj, loc, **kwargs):
    """
    Recreate a plot's legend at a new location.

    The name is a slight misnomer. Matplotlib legends do not expose public
    control over their position parameters. So this function creates a new legend,
    copying over the data from the original object, which is then removed.

    Parameters
    ----------
    obj : the object with the plot
        This argument can be either a seaborn or matplotlib object:

        - :class:`seaborn.FacetGrid` or :class:`seaborn.PairGrid`
        - :class:`matplotlib.axes.Axes` or :class:`matplotlib.figure.Figure`

    loc : str or int
        Location argument, as in :meth:`matplotlib.axes.Axes.legend`.

    kwargs
        Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.legend`.

    Examples
    --------

    .. include:: ../docstrings/move_legend.rst

    """
    pass

def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    pass

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    pass

def get_dataset_names():
    """Report available example datasets, useful for reporting issues.

    Requires an internet connection.

    """
    pass

def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is used by :func:`load_dataset`.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the `SEABORN_DATA` environment variable (if it exists)
    or otherwise default to an OS-appropriate user cache location.

    """
    pass

def load_dataset(name, cache=True, data_home=None, **kws):
    """Load an example dataset from the online repository (requires internet).

    This function provides quick access to a small number of example datasets
    that are useful for documenting seaborn or generating reproducible examples
    for bug reports. It is not necessary for normal usage.

    Note that some of the datasets have a small amount of preprocessing applied
    to define a proper ordering for categorical variables.

    Use :func:`get_dataset_names` to see a list of available datasets.

    Parameters
    ----------
    name : str
        Name of the dataset (``{name}.csv`` on
        https://github.com/mwaskom/seaborn-data).
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory in which to cache data; see :func:`get_data_home`.
    kws : keys and values, optional
        Additional keyword arguments are passed to passed through to
        :func:`pandas.read_csv`.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.

    """
    pass

def axis_ticklabels_overlap(labels):
    """Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels

    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    pass

def axes_ticklabels_overlap(ax):
    """Return booleans for whether the x and y ticklabels on an Axes overlap.

    Parameters
    ----------
    ax : matplotlib Axes

    Returns
    -------
    x_overlap, y_overlap : booleans
        True when the labels on that axis overlap.

    """
    pass

def locator_to_legend_entries(locator, limits, dtype):
    """Return levels and formatted levels for brief numeric legends."""
    pass

def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.

    Returns
    -------
    luminance : float(s) between 0 and 1

    """
    pass

def to_utf8(obj):
    """Return a string representing a Python object.

    Strings (i.e. type ``str``) are returned unchanged.

    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.

    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object

    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    pass

def _check_argument(param, options, value, prefix=False):
    """Raise if value for param is not in options."""
    pass

def _assign_default_kwargs(kws, call_func, source_func):
    """Assign default kwargs for call_func using values from source_func."""
    pass

def adjust_legend_subtitles(legend):
    """
    Make invisible-handle "subtitles" entries look more like titles.

    Note: This function is not part of the public API and may be changed or removed.

    """
    pass

def _deprecate_ci(errorbar, ci):
    """
    Warn on usage of ci= and convert to appropriate errorbar= arg.

    ci was deprecated when errorbar was added in 0.12. It should not be removed
    completely for some time, but it can be moved out of function definitions
    (and extracted from kwargs) after one cycle.

    """
    pass

def _get_transform_functions(ax, axis):
    """Return the forward and inverse transforms for a given axis."""
    pass

@contextmanager
def _disable_autolayout():
    """Context manager for preventing rc-controlled auto-layout behavior."""
    pass

def _version_predates(lib: ModuleType, version: str) -> bool:
    """Helper function for checking version compatibility."""
    pass
