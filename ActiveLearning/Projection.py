import numpy as _np


def ProjSimplex(y):
    """
    Computes the projection onto the simplex using
    Algorithm 1 of (Chen & Ye, 2011).
    """
    n = y.size
    J = _np.argsort(y)
    i = n-1
    while True:
        ti = (_np.sum(y[J][i:])-1)/(n-i)
        i -= 1
        if ti >= y[J][i]:
            tHat = ti
            break
        elif i == 0:
            tHat = (_np.sum(y)-1)/n
            break
    return _np.maximum(0, y - tHat)
