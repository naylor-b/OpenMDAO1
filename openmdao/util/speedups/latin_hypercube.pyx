
from __future__ import division
from six import itervalues

from numpy.random import randint
from numpy.linalg import norm
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t


cpdef _perturb(np.ndarray[DTYPE_t, ndim=2] doe_arr, int mutation_count):
    """ Interchanges pairs of randomly chosen elements within randomly chosen
    columns of a DOE a number of times. The result of this operation will also
    be a Latin hypercube.
    """

    new_doe_arr = doe_arr.copy()
    cdef DTYPE_t[:, :] doe = doe_arr
    cdef DTYPE_t[:, :] new_doe = new_doe_arr
    cdef DTYPE_t n, k, count, col, el1, el2
    cdef DTYPE_t nm1 = new_doe.shape[0]-1
    cdef DTYPE_t km1 = new_doe.shape[1]-1
    cdef np.ndarray[DTYPE_t,
                    ndim=1,
                    negative_indices=False,
                    mode='c'] randcols = np.random.random_integers(0, km1, km1*mutation_count)
    cdef np.ndarray[DTYPE_t,
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


cpdef mmphi(np.ndarray[DTYPE_t, ndim=2] arr, DTYPE_t q, DTYPE_t p):
    """Returns the Morris-Mitchell sampling criterion for this Latin
    hypercube.
    """

    cdef float phi

    distdict = {}

    # Calculate the norm between each pair of points in the DOE
    cdef DTYPE_t n = arr.shape[0]
    cdef DTYPE_t m = arr.shape[1]
    cdef DTYPE_t i, j, size
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

    cdef np.ndarray[double, ndim=1] distinct_d = np.fromiter(distdict, dtype=float, count=size)

    # Mutltiplicity array with a count of how many pairs of points
    # have a given distance
    cdef np.ndarray[DTYPE_t, ndim=1] J = np.fromiter(itervalues(distdict), dtype=DTYPE, count=size)

    if q == 1:
        phi = np.sum(J * (distinct_d ** (-q)))
    else:
        phi = np.sum(J * (distinct_d ** (-q))) ** (1.0 / q)

    return phi
