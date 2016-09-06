
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from __future__ import division
from six import itervalues

from numpy.random import randint, random_integers
from numpy.linalg import norm

import numpy as np
cimport numpy as np
from libc.math cimport floor, round

DTYPE_I = np.int
ctypedef np.int_t DTYPE_I_t


cdef DTYPE_I_t[:, ::1] _perturb_(DTYPE_I_t[:, ::1] doe, int mutation_count):
    """ Interchanges pairs of randomly chosen elements within randomly chosen
    columns of a DOE a number of times. The result of this operation will also
    be a Latin hypercube.
    """

    cdef:
        DTYPE_I_t[:, ::1] new_doe = doe.copy()
        DTYPE_I_t n, k, rcount, ccount, col, el1, el2
        DTYPE_I_t nrandrows = mutation_count*2*3
        DTYPE_I_t nm1 = new_doe.shape[0]-1
        DTYPE_I_t km1 = new_doe.shape[1]-1

        DTYPE_I_t[::1] randcols = random_integers(0, km1, mutation_count)
        DTYPE_I_t[::1] randrows = random_integers(0, nm1, nrandrows)

    rcount = 0
    ccount = 0

    for count in range(mutation_count):
        col = randcols[ccount]
        ccount += 1

        # Choosing two distinct random points
        el1 = randrows[rcount]
        el2 = randrows[rcount+1]
        rcount += 3

        while el1 == el2 and rcount<nrandrows:
            el2 = randrows[rcount]
            rcount += 1

        new_doe[el1, col] = doe[el2, col]
        new_doe[el2, col] = doe[el1, col]

    return new_doe

cdef void _calc_norms_p1_(DTYPE_I_t[:, ::1] arr, double[::1] norms) nogil:
    cdef DTYPE_I_t n = arr.shape[0]
    cdef DTYPE_I_t m = arr.shape[1]
    cdef DTYPE_I_t i, ii, col, ncount = 0
    cdef double sum, diff

    for i in range(1, n):
        for ii in range(i-1):
            # calculate the norm
            sum = 0.0
            for col in range(m):
                diff = arr[i, col] - arr[ii, col]
                if diff < 0.0:
                    diff = -diff
                sum += diff
            norms[ncount] = sum
            ncount += 1


cdef void _calc_norms_(DTYPE_I_t[:, ::1] arr, double[::1] norms, DTYPE_I_t p) nogil:
    cdef DTYPE_I_t n = arr.shape[0]
    cdef DTYPE_I_t m = arr.shape[1]
    cdef DTYPE_I_t i, ii, col, ncount = 0
    cdef double sum, diff

    for i in range(1, n):
        for ii in range(i-1):
            # calculate the norm
            sum = 0.0
            for col in range(m):
                diff = arr[i, col] - arr[ii, col]
                sum += diff**p
            norms[ncount] = sum**(1./p)
            ncount += 1


cdef double _mmphi_(DTYPE_I_t[:, ::1] arr, double[::1] norms,
             DTYPE_I_t q, DTYPE_I_t p):
    """Returns the Morris-Mitchell sampling criterion for this Latin
    hypercube.
    """

    # Calculate the norm between each pair of points in the DOE
    cdef DTYPE_I_t n = arr.shape[0]
    cdef DTYPE_I_t i

    # Calculate the norm between each pair of points in the DOE
    if p == 1:
        _calc_norms_p1_(arr, norms)
    else:
        _calc_norms_(arr, norms, p)

    cdef double[::1] dist
    cdef DTYPE_I_t[::1] J

    # distance array and mutltiplicity array J with a count of how many pairs of points
    # have a given distance
    dist, J = np.unique(norms, False, False, True)

    cdef double phi = 0.0
    n = dist.shape[0]

    for i in range(n):
        phi += J[i]*(dist[i]**(-q))

    if q > 1:
        phi = phi ** (1.0/q)

    return phi


cpdef mmlhs(DTYPE_I_t[:, ::1] x_start, double phi_best,
            DTYPE_I_t q, DTYPE_I_t p, DTYPE_I_t population,
            DTYPE_I_t generations):
    """Evolutionary search for most space filling Latin-Hypercube.
    Returns a new LatinHypercube instance with an optimized set of points.
    """

    cdef DTYPE_I_t[:, ::1] x_best = x_start
    cdef DTYPE_I_t n = x_start.shape[0]
    cdef DTYPE_I_t k = x_start.shape[1]

    cdef DTYPE_I_t level_off = <DTYPE_I_t>floor(0.85 * generations)
    cdef double nval = 1 + (0.5 * k - 1)
    cdef double phi_improved, phi_try

    cdef DTYPE_I_t it, mutations, offspring
    cdef DTYPE_I_t[:, ::1] x_try, x_improved

    cdef double[::1] norms = np.empty(n*(n-1)/2)

    for it in range(generations):
        if it < level_off and level_off > 1.:
            mutations = <DTYPE_I_t>(round(nval * (level_off - it) / (level_off - 1)))
        else:
            mutations = 1

        x_improved = x_best
        phi_improved = phi_best

        for offspring in range(population):
            x_try = _perturb_(x_best, mutations)
            phi_try = _mmphi_(x_try, norms, q, p)

            if phi_try < phi_improved:
                x_improved = x_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            phi_best = phi_improved
            x_best = x_improved

    return np.asarray(x_best, dtype=int), phi_best
