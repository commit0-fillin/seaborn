from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from seaborn._marks.base import Mark, Mappable, MappableBool, MappableFloat, MappableColor, MappableStyle, resolve_properties, resolve_color, document_properties

class AreaBase:
    pass

@document_properties
@dataclass
class Area(AreaBase, Mark):
    """
    A fill mark drawn from a baseline to data values.

    See also
    --------
    Band : A fill mark representing an interval between values.

    Examples
    --------
    .. include:: ../docstrings/objects.Area.rst

    """
    color: MappableColor = Mappable('C0')
    alpha: MappableFloat = Mappable(0.2)
    fill: MappableBool = Mappable(True)
    edgecolor: MappableColor = Mappable(depend='color')
    edgealpha: MappableFloat = Mappable(1)
    edgewidth: MappableFloat = Mappable(rc='patch.linewidth')
    edgestyle: MappableStyle = Mappable('-')
    baseline: MappableFloat = Mappable(0, grouping=False)

    def _draw(self, subset: DataFrame, scales: dict[str, Scale], orient: str) -> None:
        """Draw a subset of the data."""
        x = subset['x'].to_numpy()
        y = subset['y'].to_numpy()
        baseline = self._resolve(subset, 'baseline', scales)

        color = resolve_color(self, subset, scales=scales)
        edgecolor = resolve_color(self, subset, prefix='edge', scales=scales)
        fill = self._resolve(subset, 'fill', scales)
        edgewidth = self._resolve(subset, 'edgewidth', scales)
        edgestyle = self._resolve(subset, 'edgestyle', scales)

        if orient == 'y':
            x, y = y, x
            baseline = np.full_like(x, baseline)

        plt.fill_between(x, y, baseline, color=color, alpha=self.alpha.default,
                         edgecolor=edgecolor, linewidth=edgewidth, linestyle=edgestyle,
                         fill=fill)

@document_properties
@dataclass
class Band(AreaBase, Mark):
    """
    A fill mark representing an interval between values.

    See also
    --------
    Area : A fill mark drawn from a baseline to data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Band.rst

    """
    color: MappableColor = Mappable('C0')
    alpha: MappableFloat = Mappable(0.2)
    fill: MappableBool = Mappable(True)
    edgecolor: MappableColor = Mappable(depend='color')
    edgealpha: MappableFloat = Mappable(1)
    edgewidth: MappableFloat = Mappable(0)
    edgestyle: MappableStyle = Mappable('-')

    def _draw(self, subset: DataFrame, scales: dict[str, Scale], orient: str) -> None:
        """Draw a subset of the data."""
        x = subset['x'].to_numpy()
        y_low = subset['y_low'].to_numpy()
        y_high = subset['y_high'].to_numpy()

        color = resolve_color(self, subset, scales=scales)
        edgecolor = resolve_color(self, subset, prefix='edge', scales=scales)
        fill = self._resolve(subset, 'fill', scales)
        edgewidth = self._resolve(subset, 'edgewidth', scales)
        edgestyle = self._resolve(subset, 'edgestyle', scales)

        if orient == 'y':
            x, y_low, y_high = y_low, y_high, x

        plt.fill_between(x, y_low, y_high, color=color, alpha=self.alpha.default,
                         edgecolor=edgecolor, linewidth=edgewidth, linestyle=edgestyle,
                         fill=fill)
