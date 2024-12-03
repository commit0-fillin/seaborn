import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    from ipywidgets import interact, FloatSlider, IntSlider
except ImportError:
from .miscplot import palplot
from .palettes import color_palette, dark_palette, light_palette, diverging_palette, cubehelix_palette
__all__ = ['choose_colorbrewer_palette', 'choose_cubehelix_palette', 'choose_dark_palette', 'choose_light_palette', 'choose_diverging_palette']

def _init_mutable_colormap():
    """Create a matplotlib colormap that will be updated by the widgets."""
    return LinearSegmentedColormap.from_list("interactive", ["#FFFFFF", "#000000"], N=256)

def _update_lut(cmap, colors):
    """Change the LUT values in a matplotlib colormap in-place."""
    cmap._lut[:256] = colors
    cmap._lut[:, -1] = 1  # Set alpha to 1

def _show_cmap(cmap):
    """Show a continuous matplotlib colormap."""
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto')
    ax.set_axis_off()
    plt.show()

def choose_colorbrewer_palette(data_type, as_cmap=False):
    """Select a palette from the ColorBrewer set."""
    from .palettes import color_palette
    
    data_type = data_type.lower()
    if data_type.startswith("q"):
        data_type = "qualitative"
    elif data_type.startswith("s"):
        data_type = "sequential"
    elif data_type.startswith("d"):
        data_type = "diverging"
    else:
        raise ValueError("data_type must be 'sequential', 'diverging', or 'qualitative'")

    palettes = {
        "sequential": ["Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"],
        "diverging": ["RdBu", "RdGy", "PiYG", "PRGn", "BrBG", "RdYlBu"],
        "qualitative": ["Set1", "Set2", "Set3", "Paired", "Accent", "Dark2"]
    }

    def choose_palette(pal_name):
        return color_palette(pal_name, as_cmap=as_cmap)

    return interact(
        choose_palette,
        pal_name=palettes[data_type]
    )

def choose_dark_palette(input='husl', as_cmap=False):
    """Launch an interactive widget to create a dark sequential palette."""
    from .palettes import dark_palette

    def choose_dark_pal(color, n_colors, reverse):
        return dark_palette(color, n_colors, reverse=reverse, as_cmap=as_cmap, input=input)

    return interact(
        choose_dark_pal,
        color=["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"],
        n_colors=IntSlider(min=2, max=16, value=8),
        reverse=False
    )

def choose_light_palette(input='husl', as_cmap=False):
    """Launch an interactive widget to create a light sequential palette."""
    from .palettes import light_palette

    def choose_light_pal(color, n_colors, reverse):
        return light_palette(color, n_colors, reverse=reverse, as_cmap=as_cmap, input=input)

    return interact(
        choose_light_pal,
        color=["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"],
        n_colors=IntSlider(min=2, max=16, value=8),
        reverse=False
    )

def choose_diverging_palette(as_cmap=False):
    """Launch an interactive widget to choose a diverging color palette."""
    from .palettes import diverging_palette

    def choose_diverging_pal(h_neg, h_pos, s, l, sep, n, center):
        return diverging_palette(h_neg, h_pos, s=s, l=l, sep=sep, n=n, center=center, as_cmap=as_cmap)

    return interact(
        choose_diverging_pal,
        h_neg=IntSlider(min=0, max=359, value=220),
        h_pos=IntSlider(min=0, max=359, value=10),
        s=IntSlider(min=0, max=100, value=75),
        l=IntSlider(min=0, max=100, value=50),
        sep=IntSlider(min=1, max=50, value=10),
        n=IntSlider(min=2, max=16, value=8),
        center=["light", "dark"]
    )

def choose_cubehelix_palette(as_cmap=False):
    """Launch an interactive widget to create a sequential cubehelix palette."""
    from .palettes import cubehelix_palette

    def choose_cubehelix_pal(start, rot, gamma, hue, light, dark, reverse):
        return cubehelix_palette(8, start=start, rot=rot, gamma=gamma, hue=hue,
                                 light=light, dark=dark, reverse=reverse, as_cmap=as_cmap)

    return interact(
        choose_cubehelix_pal,
        start=FloatSlider(min=0, max=3, value=0),
        rot=FloatSlider(min=-1, max=1, value=0.4),
        gamma=FloatSlider(min=0, max=5, value=1),
        hue=FloatSlider(min=0, max=1, value=0.8),
        light=FloatSlider(min=0, max=1, value=0.85),
        dark=FloatSlider(min=0, max=1, value=0.15),
        reverse=False
    )
