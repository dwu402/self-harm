import numpy as np


def draw_r(r_range):
    """Does a uniform draw on the range specified by a list"""
    return np.random.uniform(r_range[0], r_range[-1])


def is_autumn(time):
    """Determines whether or not it is autumn"""
    month = np.mod(time*12, 12)
    return month >= 3 and month < 6


def model(t, x, p):
    """Returns the rate of change of an influenza model

    Uses the model proposed in "Modelling the Impacts of Climate
    Change On Infectious Diseases in New Zealand" by Tompkins et al.

    Inputs
    t (float): time
    x (list<float>[3]): state of the population
    p (list<>): parameters
        Expected parameters:
            [float, float, float, float, float,
             list<float>[2], list<float>[2], float function(float)]
    """

    # define the parameters / parameter functions
    mu = p[0]  # natural_death_rate
    eps = p[1]  # vaccine_effectiveness
    nu = p[2]  # vaccine_coverage
    gamma = p[3]  # waning_rate
    alpha = p[4]   # recovery_rate
    r_zero_min = draw_r(p[5])  # minimum basic reproduction rate
    r_zero_max = draw_r(p[6])  # maximum basic reproduction rate
    q = p[7](t)  # specific humidity

    # calculate some parameters
    beta = (r_zero_min +
            np.exp(-180*q + np.log(r_zero_max - r_zero_min))) * (mu + alpha)
    v = is_autumn(t) * eps*nu

    # extract and name the state variables
    s = x[0]
    i = x[1]
    r = x[2]

    # specify the the rate of change
    dxdt = np.empty([3])

    dxdt[0] = mu - (mu + v)*s + gamma*r - beta*s*i
    dxdt[1] = beta*s*i - (alpha + mu)*i
    dxdt[2] = alpha*i + v*s - (gamma + mu)*r

    return dxdt
