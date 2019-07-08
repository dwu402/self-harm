import numpy as np

def model(t, state, parameters):

    x, y, z = state
    r, k, p, m, s, c, d, g, j = parameters

    return [
        r*x - k*x*y,
        p*x/(m**2+x**2) + s*(y**3)/(c**3+y**3) - d*y,
        - g*z - j*x
    ]

def model_form():
    return {
        "state": 3,
        "parameters": 9,
    }
