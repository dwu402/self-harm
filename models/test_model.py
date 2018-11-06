import numpy as np


def model(x, t, p):
    dx = np.zeros(2)
    r = p[0]
    k = p[1]
    dx[0] = -x[0]/r + x[1]/k
    dx[1] = x[0]/k - x[1]/r

    return dx
