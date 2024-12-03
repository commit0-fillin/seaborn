from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import QUAL_PALETTES, color_palette
from seaborn.utils import _check_argument, _version_predates, desaturate, locator_to_legend_entries, get_color_cycle, remove_na

class SemanticMapping:
    """Base class for mapping data values to plot attributes."""
    map_type: str | None = None
    levels = None
    lookup_table = None

    def __init__(self, plotter):
        self.plotter = plotter

    def _check_list_length(self, levels, values, variable):
        """Input check when values are provided as a list."""
        if len(values) != len(levels):
            err = f"The {variable} list has {len(values)} values but there are {len(levels)} {variable} levels"
            raise ValueError(err)

    def _lookup_single(self, key):
        """Apply the mapping to a single data value."""
        try:
            return self.lookup_table[key]
        except KeyError:
            if self.map_type == "numeric":
                return self.cmap(self.norm(key))
            else:
                return None

    def __call__(self, key, *args, **kwargs):
        """Get the attribute(s) values for the data key."""
        if isinstance(key, (list, np.ndarray, pd.Series)):
            return [self._lookup_single(k, *args, **kwargs) for k in key]
        else:
            return self._lookup_single(key, *args, **kwargs)

class HueMapping(SemanticMapping):
    """Mapping that sets artist colors according to data values."""
    palette = None
    norm = None
    cmap = None

    def __init__(self, plotter, palette=None, order=None, norm=None, saturation=1):
        """Map the levels of the `hue` variable to distinct colors.

        Parameters
        ----------
        # TODO add generic parameters

        """
        super().__init__(plotter)
        data = plotter.plot_data.get('hue', pd.Series(dtype=float))
        if isinstance(palette, np.ndarray):
            msg = 'Numpy array is not a supported type for `palette`. Please convert your palette to a list. This will become an error in v0.14'
            warnings.warn(msg, stacklevel=4)
            palette = palette.tolist()
        if data.isna().all():
            if palette is not None:
                msg = 'Ignoring `palette` because no `hue` variable has been assigned.'
                warnings.warn(msg, stacklevel=4)
        else:
            map_type = self.infer_map_type(palette, norm, plotter.input_format, plotter.var_types['hue'])
            if map_type == 'numeric':
                data = pd.to_numeric(data)
                levels, lookup_table, norm, cmap = self.numeric_mapping(data, palette, norm)
            elif map_type == 'categorical':
                cmap = norm = None
                levels, lookup_table = self.categorical_mapping(data, palette, order)
            else:
                cmap = norm = None
                levels, lookup_table = self.categorical_mapping(list(data), palette, order)
            self.saturation = saturation
            self.map_type = map_type
            self.lookup_table = lookup_table
            self.palette = palette
            self.levels = levels
            self.norm = norm
            self.cmap = cmap

    def _lookup_single(self, key):
        """Get the color for a single value, using colormap to interpolate."""
        try:
            return self.lookup_table[key]
        except KeyError:
            if self.map_type == "numeric" and self.cmap is not None:
                return self.cmap(self.norm(key))
            else:
                return None

    def infer_map_type(self, palette, norm, input_format, var_type):
        """Determine how to implement the mapping."""
        if isinstance(palette, dict):
            return "categorical"
        elif norm is not None:
            return "numeric"
        elif var_type == "numeric":
            return "numeric"
        else:
            return "categorical"

    def categorical_mapping(self, data, palette, order):
        """Determine colors when the hue mapping is categorical."""
        levels = categorical_order(data, order)
        n_colors = len(levels)

        if isinstance(palette, dict):
            missing = set(levels) - set(palette)
            if any(missing):
                err = f"The palette dictionary is missing {missing}"
                raise ValueError(err)
            lookup_table = palette
        else:
            if palette is None:
                if n_colors <= len(QUAL_PALETTES['deep']):
                    colors = QUAL_PALETTES['deep']
                else:
                    colors = color_palette('husl', n_colors)
            else:
                colors = color_palette(palette, n_colors)
            lookup_table = dict(zip(levels, colors))

        return levels, lookup_table

    def numeric_mapping(self, data, palette, norm):
        """Determine colors when the hue variable is quantitative."""
        levels = remove_na(data)
        if not levels.size:
            return [], {}

        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, tuple):
            norm = mpl.colors.Normalize(*norm)
        elif not isinstance(norm, mpl.colors.Normalize):
            err = "``norm`` must be None, tuple, or Normalize object."
            raise ValueError(err)

        if palette is None:
            cmap = mpl.cm.rocket
        elif isinstance(palette, str):
            cmap = mpl.cm.get_cmap(palette)
        elif isinstance(palette, mpl.colors.Colormap):
            cmap = palette
        else:
            err = "``palette`` must be None, string, or Colormap object."
            raise ValueError(err)

        norm.clip = True
        if not norm.scaled():
            norm(np.asarray(levels))

        lookup_table = dict(zip(levels, cmap(norm(levels))))
        return levels, lookup_table, norm, cmap

class SizeMapping(SemanticMapping):
    """Mapping that sets artist sizes according to data values."""
    norm = None

    def __init__(self, plotter, sizes=None, order=None, norm=None):
        """Map the levels of the `size` variable to distinct values.

        Parameters
        ----------
        # TODO add generic parameters

        """
        super().__init__(plotter)
        data = plotter.plot_data.get('size', pd.Series(dtype=float))
        if data.notna().any():
            map_type = self.infer_map_type(norm, sizes, plotter.var_types['size'])
            if map_type == 'numeric':
                levels, lookup_table, norm, size_range = self.numeric_mapping(data, sizes, norm)
            elif map_type == 'categorical':
                levels, lookup_table = self.categorical_mapping(data, sizes, order)
                size_range = None
            else:
                levels, lookup_table = self.categorical_mapping(list(data), sizes, order)
                size_range = None
            self.map_type = map_type
            self.levels = levels
            self.norm = norm
            self.sizes = sizes
            self.size_range = size_range
            self.lookup_table = lookup_table

class StyleMapping(SemanticMapping):
    """Mapping that sets artist style according to data values."""
    map_type = 'categorical'

    def __init__(self, plotter, markers=None, dashes=None, order=None):
        """Map the levels of the `style` variable to distinct values.

        Parameters
        ----------
        # TODO add generic parameters

        """
        super().__init__(plotter)
        data = plotter.plot_data.get('style', pd.Series(dtype=float))
        if data.notna().any():
            if variable_type(data) == 'datetime':
                data = list(data)
            levels = categorical_order(data, order)
            markers = self._map_attributes(markers, levels, unique_markers(len(levels)), 'markers')
            dashes = self._map_attributes(dashes, levels, unique_dashes(len(levels)), 'dashes')
            paths = {}
            filled_markers = []
            for k, m in markers.items():
                if not isinstance(m, mpl.markers.MarkerStyle):
                    m = mpl.markers.MarkerStyle(m)
                paths[k] = m.get_path().transformed(m.get_transform())
                filled_markers.append(m.is_filled())
            if any(filled_markers) and (not all(filled_markers)):
                err = 'Filled and line art markers cannot be mixed'
                raise ValueError(err)
            lookup_table = {}
            for key in levels:
                lookup_table[key] = {}
                if markers:
                    lookup_table[key]['marker'] = markers[key]
                    lookup_table[key]['path'] = paths[key]
                if dashes:
                    lookup_table[key]['dashes'] = dashes[key]
            self.levels = levels
            self.lookup_table = lookup_table

    def _lookup_single(self, key, attr=None):
        """Get attribute(s) for a given data point."""
        try:
            value = self.lookup_table[key]
        except KeyError:
            value = {}

        if attr is None:
            return value
        elif attr in value:
            return value[attr]
        else:
            return None

    def _map_attributes(self, arg, levels, defaults, attr):
        """Handle the specification for a given style attribute."""
        if arg is None:
            return dict(zip(levels, defaults))
        elif isinstance(arg, dict):
            missing = set(levels) - set(arg)
            if any(missing):
                err = f"These `{attr}` levels are missing {missing}"
                raise ValueError(err)
            return arg
        elif isinstance(arg, Iterable) and not isinstance(arg, str):
            if len(arg) != len(levels):
                err = f"The `{attr}` list has {len(arg)} values but there are {len(levels)} levels"
                raise ValueError(err)
            return dict(zip(levels, arg))
        else:
            err = f"`{attr}` must be a dictionary or a list of values"
            raise ValueError(err)

class VectorPlotter:
    """Base class for objects underlying *plot functions."""
    wide_structure = {'x': '@index', 'y': '@values', 'hue': '@columns', 'style': '@columns'}
    flat_structure = {'x': '@index', 'y': '@values'}
    _default_size_range = (1, 2)

    def __init__(self, data=None, variables={}):
        self._var_levels = {}
        self._var_ordered = {'x': False, 'y': False}
        self.assign_variables(data, variables)
        for var in ['hue', 'size', 'style']:
            if var in variables:
                getattr(self, f'map_{var}')()

    @property
    def has_xy_data(self):
        """Return True at least one of x or y is defined."""
        return bool(self.plot_data.get("x") is not None or self.plot_data.get("y") is not None)

    @property
    def var_levels(self):
        """Property interface to ordered list of variables levels.

        Each time it's accessed, it updates the var_levels dictionary with the
        list of levels in the current semantic mappers. But it also allows the
        dictionary to persist, so it can be used to set levels by a key. This is
        used to track the list of col/row levels using an attached FacetGrid
        object, but it's kind of messy and ideally fixed by improving the
        faceting logic so it interfaces better with the modern approach to
        tracking plot variables.

        """
        for var in ["hue", "size", "style"]:
            if hasattr(self, f"{var}_map"):
                mapper = getattr(self, f"{var}_map")
                if mapper.levels is not None:
                    self._var_levels[var] = list(mapper.levels)
        return self._var_levels

    def assign_variables(self, data=None, variables={}):
        """Define plot variables, optionally using lookup from `data`."""
        pass

    def _assign_variables_wideform(self, data=None, **kwargs):
        """Define plot variables given wide-form data.

        Parameters
        ----------
        data : flat vector or collection of vectors
            Data can be a vector or mapping that is coerceable to a Series
            or a sequence- or mapping-based collection of such vectors, or a
            rectangular numpy array, or a Pandas DataFrame.
        kwargs : variable -> data mappings
            Behavior with keyword arguments is currently undefined.

        Returns
        -------
        plot_data : :class:`pandas.DataFrame`
            Long-form data object mapping seaborn variables (x, y, hue, ...)
            to data vectors.
        variables : dict
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).

        """
        pass

    def iter_data(self, grouping_vars=None, *, reverse=False, from_comp_data=False, by_facet=True, allow_empty=False, dropna=True):
        """Generator for getting subsets of data defined by semantic variables.

        Also injects "col" and "row" into grouping semantics.

        Parameters
        ----------
        grouping_vars : string or list of strings
            Semantic variables that define the subsets of data.
        reverse : bool
            If True, reverse the order of iteration.
        from_comp_data : bool
            If True, use self.comp_data rather than self.plot_data
        by_facet : bool
            If True, add faceting variables to the set of grouping variables.
        allow_empty : bool
            If True, yield an empty dataframe when no observations exist for
            combinations of grouping variables.
        dropna : bool
            If True, remove rows with missing data.

        Yields
        ------
        sub_vars : dict
            Keys are semantic names, values are the level of that semantic.
        sub_data : :class:`pandas.DataFrame`
            Subset of ``plot_data`` for this combination of semantic values.

        """
        pass

    @property
    def comp_data(self):
        """Dataframe with numeric x and y, after unit conversion and log scaling."""
        pass

    def _get_axes(self, sub_vars):
        """Return an Axes object based on existence of row/col variables."""
        pass

    def _attach(self, obj, allowed_types=None, log_scale=None):
        """Associate the plotter with an Axes manager and initialize its units.

        Parameters
        ----------
        obj : :class:`matplotlib.axes.Axes` or :class:'FacetGrid`
            Structural object that we will eventually plot onto.
        allowed_types : str or list of str
            If provided, raise when either the x or y variable does not have
            one of the declared seaborn types.
        log_scale : bool, number, or pair of bools or numbers
            If not False, set the axes to use log scaling, with the given
            base or defaulting to 10. If a tuple, interpreted as separate
            arguments for the x and y axes.

        """
        pass

    def _get_scale_transforms(self, axis):
        """Return a function implementing the scale transform (or its inverse)."""
        pass

    def _add_axis_labels(self, ax, default_x='', default_y=''):
        """Add axis labels if not present, set visibility to match ticklabels."""
        pass

    def add_legend_data(self, ax, func, common_kws=None, attrs=None, semantic_kws=None):
        """Add labeled artists to represent the different plot semantics."""
        pass

    def _update_legend_data(self, update, var, verbosity, title, title_kws, attr_names, other_props):
        """Generate legend tick values and formatted labels."""
        pass

    def scale_categorical(self, axis, order=None, formatter=None):
        """
        Enforce categorical (fixed-scale) rules for the data on given axis.

        Parameters
        ----------
        axis : "x" or "y"
            Axis of the plot to operate on.
        order : list
            Order that unique values should appear in.
        formatter : callable
            Function mapping values to a string representation.

        Returns
        -------
        self

        """
        pass

class VariableType(UserString):
    """
    Prevent comparisons elsewhere in the library from using the wrong name.

    Errors are simple assertions because users should not be able to trigger
    them. If that changes, they should be more verbose.

    """
    allowed = ('numeric', 'datetime', 'categorical')

    def __init__(self, data):
        assert data in self.allowed, data
        super().__init__(data)

    def __eq__(self, other):
        assert other in self.allowed, other
        return self.data == other

def variable_type(vector, boolean_type='numeric'):
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """
    pass

def infer_orient(x=None, y=None, orient=None, require_numeric=True):
    """Determine how the plot should be oriented based on the data.

    For historical reasons, the convention is to call a plot "horizontally"
    or "vertically" oriented based on the axis representing its dependent
    variable. Practically, this is used when determining the axis for
    numerical aggregation.

    Parameters
    ----------
    x, y : Vector data or None
        Positional data vectors for the plot.
    orient : string or None
        Specified orientation. If not None, can be "x" or "y", or otherwise
        must start with "v" or "h".
    require_numeric : bool
        If set, raise when the implied dependent variable is not numeric.

    Returns
    -------
    orient : "x" or "y"

    Raises
    ------
    ValueError: When `orient` is an unknown string.
    TypeError: When dependent variable is not numeric, with `require_numeric`

    """
    pass

def unique_dashes(n):
    """Build an arbitrarily long list of unique dash styles for lines.

    Parameters
    ----------
    n : int
        Number of unique dash specs to generate.

    Returns
    -------
    dashes : list of strings or tuples
        Valid arguments for the ``dashes`` parameter on
        :class:`matplotlib.lines.Line2D`. The first spec is a solid
        line (``""``), the remainder are sequences of long and short
        dashes.

    """
    pass

def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.

    Parameters
    ----------
    n : int
        Number of unique marker specs to generate.

    Returns
    -------
    markers : list of string or tuples
        Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
        All markers will be filled.

    """
    pass

def categorical_order(vector, order=None):
    """Return a list of unique data values.

    Determine an ordered list of levels in ``values``.

    Parameters
    ----------
    vector : list, array, Categorical, or Series
        Vector of "categorical" values
    order : list-like, optional
        Desired order of category levels to override the order determined
        from the ``values`` object.

    Returns
    -------
    order : list
        Ordered list of category levels not including null values.

    """
    pass
