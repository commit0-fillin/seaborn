from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import Locator, Formatter, AutoLocator, AutoMinorLocator, FixedLocator, LinearLocator, LogLocator, SymmetricalLogLocator, MaxNLocator, MultipleLocator, EngFormatter, FuncFormatter, LogFormatterSciNotation, ScalarFormatter, StrMethodFormatter
from matplotlib.dates import AutoDateLocator, AutoDateFormatter, ConciseDateFormatter
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from seaborn._core.plot import Plot
    from seaborn._core.properties import Property
    from numpy.typing import ArrayLike, NDArray
    TransFuncs = Tuple[Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]
    Pipeline = Sequence[Optional[Callable[[Any], Any]]]

class Scale:
    """Base class for objects that map data values to visual properties."""
    values: tuple | str | list | dict | None
    _priority: ClassVar[int]
    _pipeline: Pipeline
    _matplotlib_scale: ScaleBase
    _spacer: staticmethod
    _legend: tuple[list[Any], list[str]] | None

    def __post_init__(self):
        self._tick_params = None
        self._label_params = None
        self._legend = None

    def _finalize(self, p: Plot, axis: Axis) -> None:
        """Perform scale-specific axis tweaks after adding artists."""
        # This is a base method and doesn't need to do anything by default
        # Subclasses can override this method to perform specific finalizations
        return

    def __call__(self, data: Series) -> ArrayLike:
        trans_data: Series | NDArray | list
        scalar_data = np.isscalar(data)
        if scalar_data:
            trans_data = np.array([data])
        else:
            trans_data = data
        for func in self._pipeline:
            if func is not None:
                trans_data = func(trans_data)
        if scalar_data:
            return trans_data[0]
        else:
            return trans_data

@dataclass
class Boolean(Scale):
    """
    A scale with a discrete domain of True and False values.

    The behavior is similar to the :class:`Nominal` scale, but property
    mappings and legends will use a [True, False] ordering rather than
    a sort using numeric rules. Coordinate variables accomplish this by
    inverting axis limits so as to maintain underlying numeric positioning.
    Input data are cast to boolean values, respecting missing data.

    """
    values: tuple | list | dict | None = None
    _priority: ClassVar[int] = 3

@dataclass
class Nominal(Scale):
    """
    A categorical scale without relative importance / magnitude.
    """
    values: tuple | str | list | dict | None = None
    order: list | None = None
    _priority: ClassVar[int] = 4

    def tick(self, locator: Locator | None=None) -> Nominal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.

        Returns
        -------
        Copy of self with new tick configuration.

        """
        new = self.__class__(**self.__dict__)
        new._tick_locator = locator
        return new

    def label(self, formatter: Formatter | None=None) -> Nominal:
        """
        Configure the selection of labels for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured matplotlib formatter; other parameters will not be used.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        new = self.__class__(**self.__dict__)
        new._tick_formatter = formatter
        return new

@dataclass
class Ordinal(Scale):
    ...

@dataclass
class Discrete(Scale):
    ...

@dataclass
class ContinuousBase(Scale):
    values: tuple | str | None = None
    norm: tuple | None = None

@dataclass
class Continuous(ContinuousBase):
    """
    A numeric scale supporting norms and functional transforms.
    """
    values: tuple | str | None = None
    trans: str | TransFuncs | None = None
    _priority: ClassVar[int] = 1

    def tick(self, locator: Locator | None=None, *, at: Sequence[float] | None=None, upto: int | None=None, count: int | None=None, every: float | None=None, between: tuple[float, float] | None=None, minor: int | None=None) -> Continuous:
        """
        Configure the selection of ticks for the scale's axis or legend.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.
        at : sequence of floats
            Place ticks at these specific locations (in data units).
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.
        count : int
            Choose exactly this number of ticks, bounded by `between` or axis limits.
        every : float
            Choose locations at this interval of separation (in data units).
        between : pair of floats
            Bound upper / lower ticks when using `every` or `count`.
        minor : int
            Number of unlabeled ticks to draw between labeled "major" ticks.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        new = self.__class__(**self.__dict__)
        if locator is not None:
            new._tick_locator = locator
        elif at is not None:
            new._tick_locator = FixedLocator(at)
        elif upto is not None:
            new._tick_locator = MaxNLocator(upto)
        elif count is not None:
            new._tick_locator = LinearLocator(count)
        elif every is not None:
            new._tick_locator = MultipleLocator(every)
        
        if between is not None:
            new._tick_between = between
        
        if minor is not None:
            new._minor_locator = AutoMinorLocator(minor + 1)
        
        return new

    def label(self, formatter: Formatter | None=None, *, like: str | Callable | None=None, base: int | None | Default=default, unit: str | None=None) -> Continuous:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured formatter to use; other parameters will be ignored.
        like : str or callable
            Either a format pattern (e.g., `".2f"`), a format string with fields named
            `x` and/or `pos` (e.g., `"${x:.2f}"`), or a callable with a signature like
            `f(x: float, pos: int) -> str`. In the latter variants, `x` is passed as the
            tick value and `pos` is passed as the tick index.
        base : number
            Use log formatter (with scientific notation) having this value as the base.
            Set to `None` to override the default formatter with a log transform.
        unit : str or (str, str) tuple
            Use  SI prefixes with these units (e.g., with `unit="g"`, a tick value
            of 5000 will appear as `5 kg`). When a tuple, the first element gives the
            separator between the number and unit.

        Returns
        -------
        scale
            Copy of self with new label configuration.

        """
        new = self.__class__(**self.__dict__)
        if formatter is not None:
            new._tick_formatter = formatter
        elif like is not None:
            if isinstance(like, str):
                new._tick_formatter = StrMethodFormatter(like)
            elif callable(like):
                new._tick_formatter = FuncFormatter(like)
        elif base is not None:
            if base is default:
                new._tick_formatter = ScalarFormatter()
            else:
                new._tick_formatter = LogFormatterSciNotation(base=base)
        elif unit is not None:
            if isinstance(unit, tuple):
                new._tick_formatter = EngFormatter(unit=unit[1], sep=unit[0])
            else:
                new._tick_formatter = EngFormatter(unit=unit)
        
        return new

@dataclass
class Temporal(ContinuousBase):
    """
    A scale for date/time data.
    """
    trans = None
    _priority: ClassVar[int] = 2

    def tick(self, locator: Locator | None=None, *, upto: int | None=None) -> Temporal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.
        upto : int
            Choose "nice" locations for ticks, but do not exceed this number.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        new = self.__class__(**self.__dict__)
        if locator is not None:
            new._tick_locator = locator
        elif upto is not None:
            new._tick_locator = AutoDateLocator(maxticks=upto)
        return new

    def label(self, formatter: Formatter | None=None, *, concise: bool=False) -> Temporal:
        """
        Configure the appearance of tick labels for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured formatter to use; other parameters will be ignored.
        concise : bool
            If True, use :class:`matplotlib.dates.ConciseDateFormatter` to make
            the tick labels as compact as possible.

        Returns
        -------
        scale
            Copy of self with new label configuration.

        """
        new = self.__class__(**self.__dict__)
        if formatter is not None:
            new._tick_formatter = formatter
        elif concise:
            if not hasattr(new, '_tick_locator') or new._tick_locator is None:
                new._tick_locator = AutoDateLocator()
            new._tick_formatter = ConciseDateFormatter(new._tick_locator)
        else:
            new._tick_formatter = AutoDateFormatter(new._tick_locator)
        return new

class PseudoAxis:
    """
    Internal class implementing minimal interface equivalent to matplotlib Axis.

    Coordinate variables are typically scaled by attaching the Axis object from
    the figure where the plot will end up. Matplotlib has no similar concept of
    and axis for the other mappable variables (color, etc.), but to simplify the
    code, this object acts like an Axis and can be used to scale other variables.

    """
    axis_name = ''

    def __init__(self, scale):
        self.converter = None
        self.units = None
        self.scale = scale
        self.major = mpl.axis.Ticker()
        self.minor = mpl.axis.Ticker()
        self._data_interval = (None, None)
        scale.set_default_locators_and_formatters(self)

    def update_units(self, x):
        """Pass units to the internal converter, potentially updating its mapping."""
        if self.converter is not None:
            self.converter.update_units(x)

    def convert_units(self, x):
        """Return a numeric representation of the input data."""
        if self.converter is not None:
            return self.converter.convert(x, self.units, self)
        return x
