"""Algorithms to support fitting routines in seaborn plotting functions."""
import numpy as np
import warnings

def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and store aggregate values.

    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.

    Keyword arguments:
        n_boot : int, default=10000
            Number of iterations
        axis : int, default=None
            Will pass axis to ``func`` as a keyword argument.
        units : array, default=None
            Array of sampling unit IDs. When used the bootstrap resamples units
            and then observations within units instead of individual
            datapoints.
        func : string or callable, default="mean"
            Function to call on the args that are passed in. If string, uses as
            name of function in the numpy namespace. If nans are present in the
            data, will try to use nan-aware version of named function.
        seed : Generator | SeedSequence | RandomState | int | None
            Seed for the random number generator; useful if you want
            reproducible resamples.

    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values

    """
    # Extract keyword arguments
    n_boot = kwargs.get('n_boot', 10000)
    axis = kwargs.get('axis', None)
    units = kwargs.get('units', None)
    func = kwargs.get('func', 'mean')
    seed = kwargs.get('seed', None)

    # Set random seed
    rng = np.random.default_rng(seed)

    # Convert string function to callable
    if isinstance(func, str):
        if func.startswith('nan'):
            func = getattr(np, func)
        else:
            func = getattr(np, func)
            func = lambda *a, **kw: func(*a, **kw)

    # Ensure all input arrays have the same length
    if len(set(arg.shape[0] for arg in args)) != 1:
        raise ValueError("All input arrays must have the same length.")

    # Prepare the function keyword arguments
    func_kwargs = {'axis': axis} if axis is not None else {}

    if units is not None:
        return _structured_bootstrap(args, n_boot, units, func, func_kwargs, rng.integers)
    
    # Perform bootstrap
    n = len(args[0])
    boot_dist = []
    for _ in range(n_boot):
        resampler = rng.integers(0, n, n)
        sample = [a[resampler] for a in args]
        boot_dist.append(func(*sample, **func_kwargs))

    return np.array(boot_dist)

def _structured_bootstrap(args, n_boot, units, func, func_kwargs, integers):
    """Resample units instead of datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)

    boot_dist = []
    for _ in range(n_boot):
        resampled_units = integers(0, n_units, n_units)
        boot_sample = []
        for arg in args:
            boot_sample.append(np.concatenate([arg[units == unit] for unit in unique_units[resampled_units]]))
        boot_dist.append(func(*boot_sample, **func_kwargs))

    return np.array(boot_dist)
