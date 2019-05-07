import numpy as np

def model(t, state, parameters):

    x, y, z = state
    r, k, p, c, s, m, d, g, j = parameters

    return [
        r*x - k*x*y,
        p*x/(c**2+x**2) + s*(y**3)/(m**3+y**3) - d*y,
        - g*z - j*x
    ]

def model_form():
    return {
        "state": 3,
        "parameters": 9,
    }
