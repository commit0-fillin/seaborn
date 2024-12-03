from __future__ import annotations
from collections.abc import Generator
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from seaborn._core.plot import FacetSpec, PairSpec
    from matplotlib.figure import SubFigure

class Subplots:
    """
    Interface for creating and using matplotlib subplots based on seaborn parameters.

    Parameters
    ----------
    subplot_spec : dict
        Keyword args for :meth:`matplotlib.figure.Figure.subplots`.
    facet_spec : dict
        Parameters that control subplot faceting.
    pair_spec : dict
        Parameters that control subplot pairing.
    data : PlotData
        Data used to define figure setup.

    """

    def __init__(self, subplot_spec: dict, facet_spec: FacetSpec, pair_spec: PairSpec):
        self.subplot_spec = subplot_spec
        self._check_dimension_uniqueness(facet_spec, pair_spec)
        self._determine_grid_dimensions(facet_spec, pair_spec)
        self._handle_wrapping(facet_spec, pair_spec)
        self._determine_axis_sharing(pair_spec)

    def _check_dimension_uniqueness(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
        """Reject specs that pair and facet on (or wrap to) same figure dimension."""
        facet_dims = set(facet_spec.get("variables", {}))
        pair_dims = set(pair_spec.get("variables", {}))
        
        if facet_spec.get("wrap") and pair_spec.get("wrap"):
            raise ValueError("Cannot wrap both facet and pair plots.")
        
        if facet_dims & pair_dims:
            raise ValueError(f"Cannot use the same dimension(s) {facet_dims & pair_dims} for both faceting and pairing.")

    def _determine_grid_dimensions(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
        """Parse faceting and pairing information to define figure structure."""
        self.nrow = 1
        self.ncol = 1

        facet_vars = facet_spec.get("variables", {})
        pair_vars = pair_spec.get("variables", {})

        if "row" in facet_vars:
            self.nrow = len(facet_vars["row"])
        if "col" in facet_vars:
            self.ncol = len(facet_vars["col"])

        if "x" in pair_vars:
            self.ncol *= len(pair_vars["x"])
        if "y" in pair_vars:
            self.nrow *= len(pair_vars["y"])

        if facet_spec.get("wrap"):
            total = self.nrow * self.ncol
            self.ncol = facet_spec["wrap"]
            self.nrow = (total - 1) // self.ncol + 1

        if pair_spec.get("wrap"):
            total = self.nrow * self.ncol
            self.ncol = pair_spec["wrap"]
            self.nrow = (total - 1) // self.ncol + 1

    def _handle_wrapping(self, facet_spec: FacetSpec, pair_spec: PairSpec) -> None:
        """Update figure structure parameters based on facet/pair wrapping."""
        if facet_spec.get("wrap"):
            self.wrapped = "facet"
            self.wrap_dim = next(iter(facet_spec.get("variables", {})))
        elif pair_spec.get("wrap"):
            self.wrapped = "pair"
            self.wrap_dim = next(iter(pair_spec.get("variables", {})))
        else:
            self.wrapped = None
            self.wrap_dim = None

    def _determine_axis_sharing(self, pair_spec: PairSpec) -> None:
        """Update subplot spec with default or specified axis sharing parameters."""
        self.sharex = 'col' if 'x' in pair_spec.get("variables", {}) else True
        self.sharey = 'row' if 'y' in pair_spec.get("variables", {}) else True

        if pair_spec.get("cross", True):
            self.sharex = True
            self.sharey = True

        self.subplot_spec.update({
            'sharex': self.sharex,
            'sharey': self.sharey
        })

    def init_figure(self, pair_spec: PairSpec, pyplot: bool=False, figure_kws: dict | None=None, target: Axes | Figure | SubFigure | None=None) -> Figure:
        """Initialize matplotlib objects and add seaborn-relevant metadata."""
        if target is None:
            if pyplot:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(self.nrow, self.ncol, **self.subplot_spec, **(figure_kws or {}))
            else:
                fig = Figure(**figure_kws or {})
                axes = fig.subplots(self.nrow, self.ncol, **self.subplot_spec)
        elif isinstance(target, Axes):
            fig = target.figure
            axes = np.array([[target]])
        elif isinstance(target, (Figure, SubFigure)):
            fig = target
            axes = fig.subplots(self.nrow, self.ncol, **self.subplot_spec)
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

        if self.nrow == self.ncol == 1:
            axes = np.array([[axes]])
        elif self.nrow == 1 or self.ncol == 1:
            axes = axes.reshape(self.nrow, self.ncol)

        self._figure = fig
        self._axes = axes
        self._subplot_list = [
            {"ax": ax, "row": i, "col": j}
            for i, row in enumerate(axes)
            for j, ax in enumerate(row)
        ]

        return fig

    def __iter__(self) -> Generator[dict, None, None]:
        """Yield each subplot dictionary with Axes object and metadata."""
        yield from self._subplot_list

    def __len__(self) -> int:
        """Return the number of subplots in this figure."""
        return len(self._subplot_list)
