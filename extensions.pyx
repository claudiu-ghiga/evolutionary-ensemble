import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from cython.parallel import prange


def pairwise_errors(long[:, ::1] Y):
    cdef Py_ssize_t n_base = Y.shape[0]
    cdef Py_ssize_t n_obs = Y.shape[1]

    # Instantiate matrices and fill wit NaN.
    Q = np.empty((n_base, n_base), dtype=np.double)
    Q.fill(np.nan)
    # Rho = np.empty((n_base, n_base), dtype=np.double)
    # Rho.fill(np.nan)
    Dis = np.empty((n_base, n_base), dtype=np.double)
    Dis.fill(np.nan)

    cdef np.double_t[:, ::1] Q_view = Q
    # cdef np.double_t[:, ::1] Rho_view = Rho
    cdef np.double_t[:, ::1] Dis_view = Dis

    cdef Py_ssize_t i, j, k
    cdef double n00, n11, n01, n10
    cdef np.double_t Q_denom
    cdef np.double_t eps = 10e-8
    for i in range(n_base - 1):
        for j in range(i + 1, n_base):
            n00 = 0
            n01 = 0
            n10 = 0
            n11 = 0
            for k in range(n_obs):
                if Y[i, k] == 0 and Y[j, k] == 0:
                    n00 = n00 + 1
                elif Y[i, k] == 0 and Y[j, k] == 1:
                    n01 = n01 + 1
                elif Y[i, k] == 1 and Y[j, k] == 0:
                    n10 = n10 + 1
                elif Y[i, k] == 1 and Y[j, k] == 1:
                    n11 = n11 + 1
            Q_denom = n11 * n00 + n01 * n10
            if Q_denom < eps:
                Q_view[i, j] = 1.0
            else:
                Q_view[i, j] = ((n11 * n00 - n01 * n10) /
                                Q_denom)
            # rho_denom = sqrt((n11 + n10) * (n01 + n00) *
            #                  (n11 + n01) * (n10 + n00))
            # if rho_denom <= eps:
            #     Rho_view[i, j] = np.inf
            # else:
            #     Rho_view[i, j] = ((n11 * n00 - n01 * n10) / rho_denom)
            Dis_view[i, j] = ((n01 + n10) / (n11 + n10 + n01 + n00))
    # return Q, Rho, Dis
    return Q, Dis
