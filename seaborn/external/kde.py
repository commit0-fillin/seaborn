"""
This module was copied from the scipy project.

In the process of copying, some methods were removed because they depended on
other parts of scipy (especially on compiled components), allowing seaborn to
have a simple and pure Python implementation. These include:

- integrate_gaussian
- integrate_box
- integrate_box_1d
- integrate_kde
- logpdf
- resample

Additionally, the numpy.linalg module was substituted for scipy.linalg,
and the examples section (with doctests) was removed from the docstring

The original scipy license is copied below:

Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import numpy as np
from numpy import asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, power, atleast_1d, sum, ones, cov
from numpy import linalg
__all__ = ['gaussian_kde']

class gaussian_kde:
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.

        .. versionadded:: 1.2.0
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    evaluate
    __call__
    integrate_gaussian
    integrate_box_1d
    integrate_box
    integrate_kde
    pdf
    logpdf
    resample
    set_bandwidth
    covariance_factor

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),

    with ``neff`` the effective number of datapoints.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)

    as detailed in [5]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272

    """

    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = atleast_2d(asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError('`dataset` input should have multiple elements.')
        self.d, self.n = self.dataset.shape
        if weights is not None:
            self._weights = atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError('`weights` input should be one-dimensional.')
            if len(self._weights) != self.n:
                raise ValueError('`weights` input should be of length n')
            self._neff = 1 / sum(self._weights ** 2)
        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        pass
    __call__ = evaluate

    def scotts_factor(self):
        """Compute Scott's factor.

        Returns
        -------
        s : float
            Scott's factor.
        """
        pass

    def silverman_factor(self):
        """Compute the Silverman factor.

        Returns
        -------
        s : float
            The silverman factor.
        """
        pass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = 'Computes the coefficient (`kde.factor`) that\n        multiplies the data covariance matrix to obtain the kernel covariance\n        matrix. The default is `scotts_factor`.  A subclass can overwrite this\n        method to provide a different method, or set it through a call to\n        `kde.set_bandwidth`.'

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        """
        pass

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        pass

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        pass