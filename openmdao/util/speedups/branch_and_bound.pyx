# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from __future__ import division
from six import itervalues

from numpy.random import randint, random_integers
from numpy.linalg import norm

import numpy as np
cimport numpy as np
from libc.math cimport exp

DTYPE_I = np.int
ctypedef np.int_t DTYPE_I_t

DTYPE_F = np.float
ctypedef np.float_t DTYPE_F_t


cpdef _interval_analysis(DTYPE_F_t[::1] lb_x, DTYPE_F_t[::1] ub_x, surrogate):
    """ The module predicts the lower and upper bound of the artificial
    variable 'r' from the bounds of the design variable x r is related to x
    by the following equation:

    r_i = exp(-sum(theta_h*(x_h - x_h_i)^2))

    """

    cdef DTYPE_F_t[:, :] X = surrogate.X
    cdef DTYPE_F_t[::1] thetas = surrogate.thetas
    cdef DTYPE_I_t p = surrogate.p
    cdef DTYPE_I_t n = X.shape[0]
    cdef DTYPE_I_t k = X.shape[1]

    cdef DTYPE_F_t t1L, t1U, t2L, t2U, t3L, t3U, t4L, t4U, prod, minp, maxp

    cdef DTYPE_F_t[::1] lb_r = np.empty(n)
    cdef DTYPE_F_t[::1] ub_r = np.empty(n)

    if p == 2:
        for i in range(n):
            t4L = 0.0
            t4U = 0.0
            for h in range(k):
                t1L = lb_x[h] - X[i, h]
                t1U = ub_x[h] - X[i, h]

                minp = t1L*t1L
                maxp = minp
                prod = t1L*t1U
                if prod < minp:
                    minp = prod
                if prod > maxp:
                    maxp = prod
                prod = t1U*t1U
                if prod < minp:
                    minp = prod
                if prod > maxp:
                    maxp = prod
                if minp < 0.0:
                    minp = 0.0
                if maxp < 0.0:
                    maxp = 0.0  # Should this be min(0, minp) since it's for an upper bound???

                t2L = minp
                t2U = maxp

                t3L = -thetas[h]*t2L
                t3U = t3L
                prod = -thetas[h]*t2U
                if prod < t3L:
                    t3L = prod
                if prod > t3U:
                    t3U = prod

                t4L += t3L
                t4U += t3U

            lb_r[i] = exp(t4L)
            ub_r[i] = exp(t4U)
    else:
        print("\nWarning! Value of p should be 2. Cannot perform interval analysis")
        print("\nReturing global bound of the r variable")

    return np.asarray(lb_r), np.asarray(ub_r)


cpdef _calc_SSqr_convex(DTYPE_F_t[:] x_com, DTYPE_F_t[:, :] x_comL,
                        DTYPE_F_t[:, :] x_comU, DTYPE_F_t[:, :] xhat_comL,
                        DTYPE_F_t[:, :] xhat_comU,
                        DTYPE_F_t[:, :] X, DTYPE_F_t[:, :] R_inv, DTYPE_F_t sum_R_inv,
                        DTYPE_F_t[:] SigmaSqr, DTYPE_F_t alpha):
    """ Callback function for minimization of mean squared error."""

    cdef DTYPE_I_t n = X.shape[0]
    cdef DTYPE_I_t k = X.shape[1]
    cdef DTYPE_I_t com_n = x_com.shape[0]

    cdef DTYPE_I_t i, j
    cdef DTYPE_F_t term0, rterm0, temp, dot_r_hats, rhat, rhatL, rhatU
    cdef DTYPE_F_t[:] r = x_com.copy()

    dot_r_hats = 0.0

    for i in range(k, com_n):
        rhat = x_com[i]
        rhatL = x_comL[i,0]
        rhatU = x_comU[i,0]

        r[i-k] = rhatL + rhat*(rhatU - rhatL)
        dot_r_hats += (rhat-rhatL)*(rhat-rhatU)

    dot_r_hats *= alpha

    term0 - 0.0
    rterm0 = 0.0

    for i in range(n):
        temp = 0.0
        for j in range(n):
            temp += R_inv[i, j]*r[j]
        term0 += temp
        rterm0 += r[i]*temp

    return -SigmaSqr[0]*(1.0 - rterm0 + ((1.0 - term0)**2/sum_R_inv)) + dot_r_hats
