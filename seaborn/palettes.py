import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
__all__ = ['color_palette', 'hls_palette', 'husl_palette', 'mpl_palette', 'dark_palette', 'light_palette', 'diverging_palette', 'blend_palette', 'xkcd_palette', 'crayon_palette', 'cubehelix_palette', 'set_color_codes']
SEABORN_PALETTES = dict(deep=['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD'], deep6=['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD'], muted=['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4', '#8C613C', '#DC7EC0', '#797979', '#D5BB67', '#82C6E2'], muted6=['#4878D0', '#6ACC64', '#D65F5F', '#956CB4', '#D5BB67', '#82C6E2'], pastel=['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#DEBB9B', '#FAB0E4', '#CFCFCF', '#FFFEA3', '#B9F2F0'], pastel6=['#A1C9F4', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#FFFEA3', '#B9F2F0'], bright=['#023EFF', '#FF7C00', '#1AC938', '#E8000B', '#8B2BE2', '#9F4800', '#F14CC1', '#A3A3A3', '#FFC400', '#00D7FF'], bright6=['#023EFF', '#1AC938', '#E8000B', '#8B2BE2', '#FFC400', '#00D7FF'], dark=['#001C7F', '#B1400D', '#12711C', '#8C0800', '#591E71', '#592F0D', '#A23582', '#3C3C3C', '#B8850A', '#006374'], dark6=['#001C7F', '#12711C', '#8C0800', '#591E71', '#B8850A', '#006374'], colorblind=['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9'], colorblind6=['#0173B2', '#029E73', '#D55E00', '#CC78BC', '#ECE133', '#56B4E9'])
MPL_QUAL_PALS = {'tab10': 10, 'tab20': 20, 'tab20b': 20, 'tab20c': 20, 'Set1': 9, 'Set2': 8, 'Set3': 12, 'Accent': 8, 'Paired': 12, 'Pastel1': 9, 'Pastel2': 8, 'Dark2': 8}
QUAL_PALETTE_SIZES = MPL_QUAL_PALS.copy()
QUAL_PALETTE_SIZES.update({k: len(v) for k, v in SEABORN_PALETTES.items()})
QUAL_PALETTES = list(QUAL_PALETTE_SIZES.keys())

class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""

    def __enter__(self):
        """Open the context."""
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        from .rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        return [mpl.colors.rgb2hex(rgb) for rgb in self]

    def _repr_html_(self):
        """Rich display of the color palette in an HTML frontend."""
        s = '<div style="display: flex; justify-content: center;">'
        for color in self:
            hex_color = mpl.colors.rgb2hex(color)
            s += f'<div style="width: 30px; height: 30px; margin: 5px; background-color: {hex_color};"></div>'
        s += '</div>'
        return s

def _patch_colormap_display():
    """Simplify the rich display of matplotlib color maps in a notebook."""
    from matplotlib.colors import Colormap
    
    def _repr_png_(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig = Figure(figsize=(6, 1))
        ax = fig.add_subplot(111)
        ax.imshow([np.linspace(0, 1, 256)], aspect='auto', cmap=self)
        ax.set_axis_off()
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = canvas.get_width_height()
        return data.reshape((h, w, 4))

    Colormap._repr_png_ = _repr_png_

def color_palette(palette=None, n_colors=None, desat=None, as_cmap=False):
    """Return a list of colors or continuous colormap defining a palette."""
    if palette is None:
        palette = get_color_cycle()
    elif isinstance(palette, str):
        if palette.lower() in SEABORN_PALETTES:
            palette = SEABORN_PALETTES[palette.lower()]
        elif palette.lower() in MPL_QUAL_PALS:
            palette = mpl_palette(palette, n_colors)
        elif palette in ['hls', 'husl']:
            func = hls_palette if palette == 'hls' else husl_palette
            palette = func(n_colors)
        elif palette.startswith('ch:'):
            palette = cubehelix_palette(n_colors, **_parse_cubehelix_args(palette))
        elif palette.startswith(('light:', 'dark:', 'blend:')):
            func_name, color_spec = palette.split(':')
            func = {'light': light_palette,
                    'dark': dark_palette,
                    'blend': blend_palette}[func_name]
            palette = func(color_spec, n_colors=n_colors)
        else:
            try:
                palette = mpl.cm.get_cmap(palette)
            except ValueError:
                raise ValueError(f"'{palette}' is not a valid palette name")

    if isinstance(palette, mpl.colors.Colormap):
        if as_cmap:
            return palette
        else:
            palette = palette(np.linspace(0, 1, n_colors))

    palette = list(palette)

    if n_colors is None:
        n_colors = len(palette)

    if desat is not None:
        palette = [desaturate(c, desat) for c in palette]

    if as_cmap:
        return mpl.colors.ListedColormap(palette)

    return _ColorPalette(cycler(palette, n_colors))

def hls_palette(n_colors=6, h=0.01, l=0.6, s=0.65, as_cmap=False):
    """
    Return hues with constant lightness and saturation in the HLS system.

    The hues are evenly sampled along a circular path. The resulting palette will be
    appropriate for categorical or cyclical data.

    The `h`, `l`, and `s` values should be between 0 and 1.

    .. note::
        While the separation of the resulting colors will be mathematically
        constant, the HLS system does not construct a perceptually-uniform space,
        so their apparent intensity will vary.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    h : float
        The value of the first hue.
    l : float
        The lightness value.
    s : float
        The saturation intensity.
    as_cmap : bool
        If True, return a matplotlib colormap object.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    husl_palette : Make a palette using evenly spaced hues in the HUSL system.

    Examples
    --------
    .. include:: ../docstrings/hls_palette.rst

    """
    pass

def husl_palette(n_colors=6, h=0.01, s=0.9, l=0.65, as_cmap=False):
    """
    Return hues with constant lightness and saturation in the HUSL system.

    The hues are evenly sampled along a circular path. The resulting palette will be
    appropriate for categorical or cyclical data.

    The `h`, `l`, and `s` values should be between 0 and 1.

    This function is similar to :func:`hls_palette`, but it uses a nonlinear color
    space that is more perceptually uniform.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    h : float
        The value of the first hue.
    l : float
        The lightness value.
    s : float
        The saturation intensity.
    as_cmap : bool
        If True, return a matplotlib colormap object.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    hls_palette : Make a palette using evenly spaced hues in the HSL system.

    Examples
    --------
    .. include:: ../docstrings/husl_palette.rst

    """
    pass

def mpl_palette(name, n_colors=6, as_cmap=False):
    """
    Return a palette or colormap from the matplotlib registry.

    For continuous palettes, evenly-spaced discrete samples are chosen while
    excluding the minimum and maximum value in the colormap to provide better
    contrast at the extremes.

    For qualitative palettes (e.g. those from colorbrewer), exact values are
    indexed (rather than interpolated), but fewer than `n_colors` can be returned
    if the palette does not define that many.

    Parameters
    ----------
    name : string
        Name of the palette. This should be a named matplotlib colormap.
    n_colors : int
        Number of discrete colors in the palette.

    Returns
    -------
    list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    Examples
    --------
    .. include:: ../docstrings/mpl_palette.rst

    """
    pass

def _color_to_rgb(color, input):
    """Add some more flexibility to color choices."""
    pass

def dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input='rgb'):
    """Make a sequential palette that blends from dark to ``color``.

    This kind of palette is good for data that range between relatively
    uninteresting low values and interesting high values.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_dark_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    light_palette : Create a sequential palette with bright low values.
    diverging_palette : Create a diverging palette with two colors.

    Examples
    --------
    .. include:: ../docstrings/dark_palette.rst

    """
    pass

def light_palette(color, n_colors=6, reverse=False, as_cmap=False, input='rgb'):
    """Make a sequential palette that blends from light to ``color``.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using a Jupyter notebook, you can also choose this palette
    interactively with the :func:`choose_light_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex code, html color name, or tuple in `input` space.
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    diverging_palette : Create a diverging palette with two colors.

    Examples
    --------
    .. include:: ../docstrings/light_palette.rst

    """
    pass

def diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center='light', as_cmap=False):
    """Make a diverging palette between two HUSL colors.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_diverging_palette` function.

    Parameters
    ----------
    h_neg, h_pos : float in [0, 359]
        Anchor hues for negative and positive extents of the map.
    s : float in [0, 100], optional
        Anchor saturation for both extents of the map.
    l : float in [0, 100], optional
        Anchor lightness for both extents of the map.
    sep : int, optional
        Size of the intermediate region.
    n : int, optional
        Number of colors in the palette (if not returning a cmap)
    center : {"light", "dark"}, optional
        Whether the center of the palette is light or dark
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark values.
    light_palette : Create a sequential palette with light values.

    Examples
    --------
    .. include: ../docstrings/diverging_palette.rst

    """
    pass

def blend_palette(colors, n_colors=6, as_cmap=False, input='rgb'):
    """Make a palette that blends between a list of colors.

    Parameters
    ----------
    colors : sequence of colors in various formats interpreted by `input`
        hex code, html color name, or tuple in `input` space.
    n_colors : int, optional
        Number of colors in the palette.
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    Examples
    --------
    .. include: ../docstrings/blend_palette.rst

    """
    pass

def xkcd_palette(colors):
    """Make a palette with color names from the xkcd color survey.

    See xkcd for the full list of colors: https://xkcd.com/color/rgb/

    This is just a simple wrapper around the `seaborn.xkcd_rgb` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.xkcd_rgb` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    crayon_palette : Make a palette with Crayola crayon colors.

    """
    pass

def crayon_palette(colors):
    """Make a palette with color names from Crayola crayons.

    Colors are taken from here:
    https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors

    This is just a simple wrapper around the `seaborn.crayons` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.crayons` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    xkcd_palette : Make a palette with named colors from the XKCD color survey.

    """
    pass

def cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=False):
    """Make a sequential palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.  "cubehelix" is
    also available as a matplotlib-based palette, but this function gives the
    user more control over the look of the palette and has a different set of
    defaults.

    In addition to using this function, it is also possible to generate a
    cubehelix palette generally in seaborn using a string starting with
    `ch:` and containing other parameters (e.g. `"ch:s=.25,r=-.5"`).

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    start : float, 0 <= start <= 3
        The hue value at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float 0 <= gamma
        Nonlinearity to emphasize dark (gamma < 1) or light (gamma > 1) colors.
    hue : float, 0 <= hue <= 1
        Saturation of the colors.
    dark : float 0 <= dark <= 1
        Intensity of the darkest color in the palette.
    light : float 0 <= light <= 1
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.
    as_cmap : bool
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    choose_cubehelix_palette : Launch an interactive widget to select cubehelix
                               palette parameters.
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.

    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    Examples
    --------
    .. include:: ../docstrings/cubehelix_palette.rst

    """
    pass

def _parse_cubehelix_args(argstr):
    """Turn stringified cubehelix params into args/kwargs."""
    pass

def set_color_codes(palette='deep'):
    """Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : {deep, muted, pastel, dark, bright, colorblind}
        Named seaborn palette to use as the source of colors.

    See Also
    --------
    set : Color codes can be set through the high-level seaborn style
          manager.
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.

    """
    pass
