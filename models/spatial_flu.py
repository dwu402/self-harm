import numpy as np

# a 3 by 3 grid of SIR with diffusion dynamics, vital dynamics, vaccination and waning immunity
# modelled as

# dS/dt = mu - mu*S - beta*S*I + gamma*R + Ds * Nabla^2 S
# dI/dt = beta*S*I - alpha*I - mu*I + Di * Nabla^2 I
# dR/dt = alpha*I - gamma*R - mu*R + Dr * Nabla^2 R

# Neumann boundaries (Nabla u on boundaries = 0)

def model(t, state, parameters):

    S = state[:9]
    I = state[9:18]
    R = state[18:]
    mu, r0, gamma, v, alpha, ds, di, dr = parameters

    # using ghost point method for applying Neumann conditions
    D = np.array([
        [-4, 2, 0, 2, 0, 0, 0, 0, 0],
        [1, -4, 1, 0, 2, 0, 0, 0, 0],
        [0, 2, -4, 0, 0, 2, 0, 0, 0],
        [1, 0, 0, -4, 2, 0, 1, 0, 0],
        [0, 1, 0, 1, -4, 1, 0, 1, 0],
        [0, 0, 1, 0, 2, -4, 0, 0, 1],
        [0, 0, 0, 2, 0, 0, -4, 2, 0],
        [0, 0, 0, 0, 2, 0, 1, -4, 1],
        [0, 0, 0, 0, 0, 2, 0, 2, -4]
    ])

    return np.array([
        mu*(S+I+R) - mu*S - r0*(mu+alpha)*S*I/(S+I+R) + gamma*R - v*S + ds*D@S,
        r0*(mu+alpha)*S*I/(S+I+R) - alpha*I - mu*I + di*D@I,
        alpha*I + v*S - gamma*R - mu*R + dr*D@R,
    ]).flatten()

def model_form():
    return {
        "state": 27,
        "parameters": 8,
    }
