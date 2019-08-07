import numpy as np

# a n by n grid of SIR with diffusion dynamics, vital dynamics, vaccination and waning immunity
# modelled as

# dS/dt = mu - mu*S - beta*S*I + gamma*R + Ds * Nabla^2 S
# dI/dt = beta*S*I - alpha*I - mu*I + Di * Nabla^2 I
# dR/dt = alpha*I - gamma*R - mu*R + Dr * Nabla^2 R

# No flux boundaries (Nabla u on boundaries = 0)

N = 21

# Build diffusion matrix
D = np.diag([-4]*(N**2))
# points above, below, right, left
D += np.diag([1]*(N**2-N), k=N)
D += np.diag([1]*(N**2-N), k=-N)
D += np.diag([1]*(N**2-1), k=1)
D += np.diag([1]*(N**2-1), k=-1)
# boundaries bottom, top, left (x2), right (x2)
D[np.arange(N), np.arange(N, 2*N)] = 2
D[np.arange(N**2-N, N**2), np.arange(N**2-N*2, N**2-N)] = 2
D[np.arange(0, N**2, N), np.arange(1, N**2, N)] = 2
D[np.arange(N, N**2, N), np.arange(N-1, N**2-1, N)] = 0
D[np.arange(N-1, N**2, N), np.arange(N-2, N**2, N)] = 2
D[np.arange(N-1, N**2-N, N), np.arange(N, N**2, N)] = 0

def model(t, state, parameters):

    S = state[:N**2]
    I = state[N**2:2*N**2]
    R = state[2*N**2:]
    mu, r0, gamma, v, alpha, ds, di, dr = parameters

    return np.array([
        mu - mu*S - r0*(mu+alpha)*S*I + gamma*R - v*S + ds*D@S,
        r0*(mu+alpha)*S*I - alpha*I - mu*I + di*D@I,
        alpha*I + v*S - gamma*R - mu*R + dr*D@R,
    ]).flatten()

def model_form():
    return {
        "state": 3*N**2,
        "parameters": 8,
    }

def default_ic():
    infected_fraction = 1e-5
    centre = int(N//2)
    s = [1]*N**2
    i = [0]*N**2
    r = [0]*N**2
    s[centre + N*centre] = 1-infected_fraction
    i[centre + N*centre] = infected_fraction

    return np.hstack([s, i, r])