
from __future__ import division
from six import itervalues

from numpy.random import randint
from numpy.linalg import norm
import numpy as np
cimport numpy as np

DTYPE_I = np.int
ctypedef np.int_t DTYPE_I_t


cdef _perturb(np.ndarray[DTYPE_I_t, ndim=2] doe_arr, int mutation_count):
    """ Interchanges pairs of randomly chosen elements within randomly chosen
    columns of a DOE a number of times. The result of this operation will also
    be a Latin hypercube.
    """

    new_doe_arr = doe_arr.copy()
    cdef:
        DTYPE_I_t[:, :] doe = doe_arr
        DTYPE_I_t[:, :] new_doe = new_doe_arr
        DTYPE_I_t n, k, count, col, el1, el2
        DTYPE_I_t nm1 = new_doe.shape[0]-1
        DTYPE_I_t km1 = new_doe.shape[1]-1
        np.ndarray[DTYPE_I_t,
                    ndim=1,
                    negative_indices=False,
                    mode='c'] randcols = np.random.random_integers(0, km1, km1*mutation_count)
        np.ndarray[DTYPE_I_t,
                    ndim=1,
                    negative_indices=False,
                    mode='c'] randrows = np.random.random_integers(0, nm1, nm1*2*mutation_count)

    for count in range(mutation_count):
        col = randcols[count]

        # Choosing two distinct random points
        el1 = randrows[count]
        el2 = randrows[count+mutation_count]
        k = 1
        while el1 == el2 and k<=nm1:
            el2 = randrows[count+mutation_count+k]
            k += 1

        new_doe[el1, col] = doe[el2, col]
        new_doe[el2, col] = doe[el1, col]

    return new_doe_arr


cdef mmphi(np.ndarray[DTYPE_I_t, ndim=2] arr, DTYPE_I_t q, DTYPE_I_t p):
    """Returns the Morris-Mitchell sampling criterion for this Latin
    hypercube.
    """

    cdef double phi

    distdict = {}

    # Calculate the norm between each pair of points in the DOE
    cdef DTYPE_I_t n = arr.shape[0]
    cdef DTYPE_I_t m = arr.shape[1]
    cdef DTYPE_I_t i, j, size
    cdef double[::1] nrm = np.empty(n, dtype=float)


    for i in range(1, n):
        nrm = norm(arr[i] - arr[:i], ord=p, axis=1)
        for j in range(i):
            nrmj = nrm[j]
            if nrmj in distdict:
                distdict[nrmj] += 1
            else:
                distdict[nrmj] = 1

    size = len(distdict)

    cdef np.ndarray[double, ndim=1] distinct_d = np.array(distdict.keys(), dtype=float)

    # Mutltiplicity array with a count of how many pairs of points
    # have a given distance
    cdef np.ndarray[DTYPE_I_t, ndim=1] J = np.array(distdict.values(), dtype=DTYPE_I)

    if q == 1:
        phi = np.sum(J * (distinct_d ** (-q)))
    else:
        phi = np.sum(J * (distinct_d ** (-q))) ** (1.0 / q)

    return phi


cpdef mmlhs(x_start, double phi_best, DTYPE_I_t q, DTYPE_I_t p, DTYPE_I_t population, DTYPE_I_t generations):
    """Evolutionary search for most space filling Latin-Hypercube.
    Returns a new LatinHypercube instance with an optimized set of points.
    """

    x_best = x_start
    cdef DTYPE_I_t n = x_start.shape[1]

    cdef DTYPE_I_t level_off = np.floor(0.85 * generations)
    cdef double nval = 1 + (0.5 * n - 1)
    cdef double phi_improved, phi_try

    cdef DTYPE_I_t it, mutations, offspring

    for it in range(generations):
        if it < level_off and level_off > 1.:
            mutations = int(round(nval * (level_off - it) / (level_off - 1)))
        else:
            mutations = 1

        x_improved = x_best
        phi_improved = phi_best

        for offspring in range(population):
            x_try = _perturb(x_best, mutations)
            phi_try = mmphi(x_try, q, p)

            if phi_try < phi_improved:
                x_improved = x_try
                phi_improved = phi_try

        if phi_improved < phi_best:
            phi_best = phi_improved
            x_best = x_improved

    return x_best, phi_best
