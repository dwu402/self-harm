#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def roots(n, b):
    p = np.zeros(n+1)
    p[0] += 1
    p[1] += -b
    p[n] += 1
    return np.roots(p)

def get_and_zip_roots(b):
    rs = roots(3, b)
    ret = []
    for r in rs:
        if np.imag(r) == 0:
            ret.append((3, np.real(r)))
    return ret

b = 20

roots = np.array(get_and_zip_roots(b))
plt.plot(roots[:,0], roots[:,1],'o')
plt.show()
