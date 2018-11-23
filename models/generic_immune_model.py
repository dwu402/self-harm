import numpy as np


def model(time, state, parameters):
    """Returns the value of the vector field at one location

    Model for a generic immune response and health recovery
    Inputs:
    time: time
    state: state vector
    parameters: parameters (r,k,p,u,v,s,n,f,g,h,j,l)
    """
    dxdt = np.empty(3)

    x, y, z = state
    r, k, p, u, v, s, n, f, g, h, j, l = parameters

    dxdt[0] = r*x - k*x*y
    dxdt[1] = p*(x**u)/(1+x**v) + s*(y**n)/(1+y**n) - y
    dxdt[2] = f*y*z - g*z - h*x*z - j*x - l*x*y
