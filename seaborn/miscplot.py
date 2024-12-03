import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
__all__ = ['palplot', 'dogplot']

def palplot(pal, size=1):
    """Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot

    """
    n = len(pal)
    fig, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0, width=0, which='major')
    ax.tick_params(axis='both', length=0, width=0, which='minor')
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.show()

def dogplot(*_, **__):
    """Who's a good boy?"""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "🐶 Woof!", fontsize=40, ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()
