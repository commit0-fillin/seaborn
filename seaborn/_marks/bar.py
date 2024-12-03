from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import Mark, Mappable, MappableBool, MappableColor, MappableFloat, MappableStyle, resolve_properties, resolve_color, document_properties
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from matplotlib.artist import Artist
    from seaborn._core.scales import Scale

class BarBase(Mark):
    def _draw(self, subset: DataFrame, scales: dict[str, Scale], orient: str) -> None:
        x = self._resolve(subset, "x", scales)
        y = self._resolve(subset, "y", scales)
        width = self._resolve(subset, "width", scales)
        color = resolve_color(self, subset, scales=scales)
        edgecolor = resolve_color(self, subset, prefix="edge", scales=scales)
        
        if orient == "v":
            self._draw_bars(x, y, width, color, edgecolor, orient)
        else:
            self._draw_bars(y, x, width, color, edgecolor, orient)

    def _draw_bars(self, pos, height, width, color, edgecolor, orient):
        raise NotImplementedError("Subclasses must implement the _draw_bars method.")

@document_properties
@dataclass
class Bar(BarBase):
    """
    A bar mark drawn between baseline and data values.

    See also
    --------
    Bars : A faster bar mark with defaults more suitable for histograms.

    Examples
    --------
    .. include:: ../docstrings/objects.Bar.rst

    """
    color: MappableColor = Mappable('C0', grouping=False)
    alpha: MappableFloat = Mappable(0.7, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(depend='color', grouping=False)
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    edgewidth: MappableFloat = Mappable(rc='patch.linewidth', grouping=False)
    edgestyle: MappableStyle = Mappable('-', grouping=False)
    width: MappableFloat = Mappable(0.8, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)

    def _draw_bars(self, pos, height, width, color, edgecolor, orient):
        import matplotlib.pyplot as plt

        baseline = self._resolve({"baseline": self.baseline}, "baseline")
        fill = self._resolve({"fill": self.fill}, "fill")
        edgewidth = self._resolve({"edgewidth": self.edgewidth}, "edgewidth")
        edgestyle = self._resolve({"edgestyle": self.edgestyle}, "edgestyle")

        if orient == "v":
            plt.bar(pos, height - baseline, width, baseline, color=color, edgecolor=edgecolor,
                    fill=fill, linewidth=edgewidth, linestyle=edgestyle)
        else:
            plt.barh(pos, height - baseline, width, baseline, color=color, edgecolor=edgecolor,
                     fill=fill, linewidth=edgewidth, linestyle=edgestyle)

@document_properties
@dataclass
class Bars(BarBase):
    """
    A faster bar mark with defaults more suitable for histograms.

    See also
    --------
    Bar : A bar mark drawn between baseline and data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Bars.rst

    """
    color: MappableColor = Mappable('C0', grouping=False)
    alpha: MappableFloat = Mappable(0.7, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(rc='patch.edgecolor', grouping=False)
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    edgewidth: MappableFloat = Mappable(auto=True, grouping=False)
    edgestyle: MappableStyle = Mappable('-', grouping=False)
    width: MappableFloat = Mappable(1, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)

    def _draw_bars(self, pos, height, width, color, edgecolor, orient):
        import matplotlib.pyplot as plt
        import numpy as np

        baseline = self._resolve({"baseline": self.baseline}, "baseline")
        fill = self._resolve({"fill": self.fill}, "fill")
        edgewidth = self._resolve({"edgewidth": self.edgewidth}, "edgewidth")
        edgestyle = self._resolve({"edgestyle": self.edgestyle}, "edgestyle")

        # Use numpy for faster calculations
        left = pos - width / 2
        right = pos + width / 2
        bottom = np.minimum(height, baseline)
        top = np.maximum(height, baseline)

        if orient == "v":
            plt.vlines(pos, bottom, top, colors=color, linewidths=width)
            if fill:
                plt.fill_between(pos, bottom, top, color=color, edgecolor=edgecolor,
                                 linewidth=edgewidth, linestyle=edgestyle)
        else:
            plt.hlines(pos, bottom, top, colors=color, linewidths=width)
            if fill:
                plt.fill_betweenx(pos, bottom, top, color=color, edgecolor=edgecolor,
                                  linewidth=edgewidth, linestyle=edgestyle)
