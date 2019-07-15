import numpy as np
import pandas as pd
from scipy import optimize
import casadi as ca

# humidity_data = pd.read_csv("data/auto-got-humidity-data.csv")

# for now use a sinusoid fit of Auckland mean humidity data
HUMIDITY_DATA = (np.array((79.3, 79.8, 80.3, 83.0, 85.8, 89.8, 88.9, 86.2, 81.3, 78.5, 77.2, 77.6)) *
                 2.541e6 * np.exp(-5415.0 / 291.15)* 18/29 / 100)
Q0 = HUMIDITY_DATA[0]
HUMIDITY_FUNCTION = lambda t, a, c, d: a*np.sin(2*np.pi*t-c)+d
COEFS, _ = optimize.curve_fit(HUMIDITY_FUNCTION,
                              np.linspace(0, 1, 12), HUMIDITY_DATA,
                              p0=[np.ptp(HUMIDITY_DATA), 0, Q0])
DQ = lambda t: 2*COEFS[0]*np.pi*np.cos(2*np.pi*t-COEFS[1])


def model(t, state, parameters):

    s, i, r, q, l = state
    mu, gm, rmin, rmax, al, v, a = parameters
    # vacc = lambda t: v if t%1 > 0.25 and t%1 < 0.5 else 0
    vacc = lambda t: 10/4*v*ca.mod(t,1)*(1-ca.mod(t, 1)**2)**4
    # vacc = lambda t: 0.25*v
    lm = rmin + (rmax-rmin)*np.exp(-a*q)
    return [
        mu + gm*r - (mu + lm*i + vacc(t))*s,
        lm*i*s - (mu + al)*i,
        al*i + vacc(t)*s - (mu + gm)*r,
        DQ(t),
        lm - l,
    ]

def model_form():
    return {
        "state": 5,
        "parameters": 7,
    }

def assumptions():
    return "\n".join([
        "No change in total population size",
        "Vaccination only happens in autumn",
        "R0 has a functional relationship with humidity as in the lm calculation",

    ])
