from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, cast
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
from pandas import DataFrame
from seaborn._core.scales import Scale
from seaborn._core.groupby import GroupBy
from seaborn._stats.base import Stat
from seaborn.utils import _version_predates
_MethodKind = Literal['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest']

@dataclass
class Perc(Stat):
    """
    Replace observations with percentile values.

    Parameters
    ----------
    k : list of numbers or int
        If a list of numbers, this gives the percentiles (in [0, 100]) to compute.
        If an integer, compute `k` evenly-spaced percentiles between 0 and 100.
        For example, `k=5` computes the 0, 25, 50, 75, and 100th percentiles.
    method : str
        Method for interpolating percentiles between observed datapoints.
        See :func:`numpy.percentile` for valid options and more information.

    Examples
    --------
    .. include:: ../docstrings/objects.Perc.rst

    """
    k: int | list[float] = 5
    method: str = 'linear'
    group_by_orient: ClassVar[bool] = True

    def __call__(self, data: DataFrame, groupby: GroupBy, orient: str, scales: dict[str, Scale]) -> DataFrame:
        var = {'x': 'y', 'y': 'x'}[orient]
        return groupby.apply(data, self._percentile, var)