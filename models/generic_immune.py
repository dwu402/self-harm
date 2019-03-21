import numpy as np

def model(t, state, parameters):

    x, y, z = state
    r, k, p, s, d, f, g, j, l = parameters

    return [
        r*x - k*x*y,
        p*x/(1+x**2) + s*(y**3)/(1+y**3) - d*y,
        f*y*z - g*z - j*x - l*y
    ]

def model_form():
    return {
        "state": 3,
        "parameters": 9,
    }
