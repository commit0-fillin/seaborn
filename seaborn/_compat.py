from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.figure import Figure
from seaborn.utils import _version_predates

def norm_from_scale(scale, norm):
    """Produce a Normalize object given a Scale and min/max domain limits."""
    if isinstance(scale, mpl.scale.LogScale):
        return mpl.colors.LogNorm(vmin=norm.vmin, vmax=norm.vmax)
    elif isinstance(scale, mpl.scale.SymmetricalLogScale):
        return mpl.colors.SymLogNorm(linthresh=scale.linthresh, linscale=scale.linscale,
                                     vmin=norm.vmin, vmax=norm.vmax)
    else:
        return mpl.colors.Normalize(vmin=norm.vmin, vmax=norm.vmax)

def get_colormap(name):
    """Handle changes to matplotlib colormap interface in 3.6."""
    if _version_predates(mpl, "3.6"):
        return mpl.cm.get_cmap(name)
    else:
        return mpl.colormaps[name]

def register_colormap(name, cmap):
    """Handle changes to matplotlib colormap interface in 3.6."""
    if _version_predates(mpl, "3.6"):
        mpl.cm.register_cmap(name, cmap)
    else:
        mpl.colormaps.register(cmap, name=name)

def set_layout_engine(fig: Figure, engine: Literal['constrained', 'compressed', 'tight', 'none']) -> None:
    """Handle changes to auto layout engine interface in 3.6"""
    if _version_predates(mpl, "3.6"):
        if engine == 'none':
            fig.set_tight_layout(False)
        elif engine == 'tight':
            fig.set_tight_layout(True)
        else:
            fig.set_constrained_layout(True)
    else:
        fig.set_layout_engine(engine)

def get_layout_engine(fig: Figure) -> mpl.layout_engine.LayoutEngine | None:
    """Handle changes to auto layout engine interface in 3.6"""
    if _version_predates(mpl, "3.6"):
        if fig.get_tight_layout():
            return mpl.layout_engine.TightLayoutEngine()
        elif fig.get_constrained_layout():
            return mpl.layout_engine.ConstrainedLayoutEngine()
        else:
            return None
    else:
        return fig.get_layout_engine()

def share_axis(ax0, ax1, which):
    """Handle changes to post-hoc axis sharing."""
    if _version_predates(mpl, "3.5"):
        if which == "x":
            ax0.get_shared_x_axes().join(ax0, ax1)
        elif which == "y":
            ax0.get_shared_y_axes().join(ax0, ax1)
    else:
        if which == "x":
            ax0.sharex(ax1)
        elif which == "y":
            ax0.sharey(ax1)

def get_legend_handles(legend):
    """Handle legendHandles attribute rename."""
    if _version_predates(mpl, "3.7"):
        return legend.legendHandles
    else:
        return legend.legend_handles
