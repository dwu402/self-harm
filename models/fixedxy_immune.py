import numpy as np

def model(t, state, parameters):

    x, y, z = state
    f,g,j,l = parameters
    r,k,p,m,s,c,d = [1.27906893e+00, 1.16352725e+00, 9.41391478e-01, 1.26082489e+00, 4.45031026e-01, 1.34197758e-03, 6.52432071e-02]
    return [
        r*x - k*x*y,
        p*x/(c**2+x**2) + s*(y**3)/(m**3+y**3) - d*y,
        f*y*z - g*z - j*x - l*y
    ]

def model_form():
    return {
        "state": 3,
        "parameters": 4,
    }
