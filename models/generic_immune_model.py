import numpy as np


def model(time, state, parameters):
    """Returns the value of the vector field at one location

    Model for a generic immune response and health recovery
    Inputs:
    time: time
    state: state vector
    parameters: parameters (r,k,p,s,d,f,g,h,j,l)
    """
    dxdt = np.empty(3)

    x, y, z = state
    r, k, p, m, s, c, d, f, g, j, l = np.abs(parameters)
    dxdt[0] = r*x - k*x*y
    dxdt[1] = p*x/(m**2+x**2) + s*(y**3)/(c**3+y**3) - d*y
    dxdt[2] = f*y*z - g*z - j*x - l*y
    return dxdt
