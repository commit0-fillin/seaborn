from __future__ import annotations
from dataclasses import dataclass, fields, field
import textwrap
from typing import Any, Callable, Union
from collections.abc import Generator
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import ndarray
from pandas import DataFrame
from matplotlib.artist import Artist
from seaborn._core.scales import Scale
from seaborn._core.properties import PROPERTIES, Property, RGBATuple, DashPattern, DashPatternWithOffset
from seaborn._core.exceptions import PlotSpecError

class Mappable:

    def __init__(self, val: Any=None, depend: str | None=None, rc: str | None=None, auto: bool=False, grouping: bool=True):
        """
        Property that can be mapped from data or set directly, with flexible defaults.

        Parameters
        ----------
        val : Any
            Use this value as the default.
        depend : str
            Use the value of this feature as the default.
        rc : str
            Use the value of this rcParam as the default.
        auto : bool
            The default value will depend on other parameters at compile time.
        grouping : bool
            If True, use the mapped variable to define groups.

        """
        if depend is not None:
            assert depend in PROPERTIES
        if rc is not None:
            assert rc in mpl.rcParams
        self._val = val
        self._rc = rc
        self._depend = depend
        self._auto = auto
        self._grouping = grouping

    def __repr__(self):
        """Nice formatting for when object appears in Mark init signature."""
        if self._val is not None:
            s = f'<{repr(self._val)}>'
        elif self._depend is not None:
            s = f'<depend:{self._depend}>'
        elif self._rc is not None:
            s = f'<rc:{self._rc}>'
        elif self._auto:
            s = '<auto>'
        else:
            s = '<undefined>'
        return s

    @property
    def depend(self) -> Any:
        """Return the name of the feature to source a default value from."""
        pass

    @property
    def default(self) -> Any:
        """Get the default value for this feature, or access the relevant rcParam."""
        pass
MappableBool = Union[bool, Mappable]
MappableString = Union[str, Mappable]
MappableFloat = Union[float, Mappable]
MappableColor = Union[str, tuple, Mappable]
MappableStyle = Union[str, DashPattern, DashPatternWithOffset, Mappable]

@dataclass
class Mark:
    """Base class for objects that visually represent data."""
    artist_kws: dict = field(default_factory=dict)

    def _resolve(self, data: DataFrame | dict[str, Any], name: str, scales: dict[str, Scale] | None=None) -> Any:
        """Obtain default, specified, or mapped value for a named feature.

        Parameters
        ----------
        data : DataFrame or dict with scalar values
            Container with data values for features that will be semantically mapped.
        name : string
            Identity of the feature / semantic.
        scales: dict
            Mapping from variable to corresponding scale object.

        Returns
        -------
        value or array of values
            Outer return type depends on whether `data` is a dict (implying that
            we want a single value) or DataFrame (implying that we want an array
            of values with matching length).

        """
        pass

    def _plot(self, split_generator: Callable[[], Generator], scales: dict[str, Scale], orient: str) -> None:
        """Main interface for creating a plot."""
        pass

def resolve_color(mark: Mark, data: DataFrame | dict, prefix: str='', scales: dict[str, Scale] | None=None) -> RGBATuple | ndarray:
    """
    Obtain a default, specified, or mapped value for a color feature.

    This method exists separately to support the relationship between a
    color and its corresponding alpha. We want to respect alpha values that
    are passed in specified (or mapped) color values but also make use of a
    separate `alpha` variable, which can be mapped. This approach may also
    be extended to support mapping of specific color channels (i.e.
    luminance, chroma) in the future.

    Parameters
    ----------
    mark :
        Mark with the color property.
    data :
        Container with data values for features that will be semantically mapped.
    prefix :
        Support "color", "fillcolor", etc.

    """
    pass