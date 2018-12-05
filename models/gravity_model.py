import numpy as np

def model(time, state, parameters):
    """Defines a gravtiy-based model for a 3 node system

    Model is an SEIR model, so each node has 4 state variables.
    Parameters are:
        beta [1x3] - infectivity by location
        alpha - transition rate from exposed to infectious
        recover [1x3] - recovery rate by location
        delta - death reate from infection
        kappa [1x3] - tunable parameter, ratio between local and distant infection effects
        distance [3x3] - distance between the nodes
        population [1x3] - population of each node
    """

    # define the population
    state_location = [state[0:4], state[4:8], state[8:12]]
    susceptibles = [state_location[x][0] for x in range(3)]
    exposed = [state_location[x][1] for x in range(3)]
    infectious = [state_location[x][2] for x in range(3)]
    recovered = [state_location[x][3] for x in range(3)]

    # define the parameters
    beta, alpha, recover, delta, kappa, distance, population = parameters

    dxdt = np.empty(4*3)

    for loc in range(3):
        dxdt[loc*4] = -beta[loc]*infectious[loc]
        for otherloc in set(range(3)).difference({loc}):
            dxdt[loc*4] -= kappa[loc]*population[loc]*population[otherloc]/distance[loc][otherloc]*beta[loc]*infectious[otherloc]
        dxdt[4*loc] *= susceptibles[loc]
        dxdt[1+loc*4] = -dxdt[loc*4] - alpha*exposed[loc]
        dxdt[2+loc*4] = alpha*exposed[loc] - recover[loc]*infectious[loc] - delta*infectious[loc]
        dxdt[3+loc*4] = recover[loc]*infectious[loc]

    return dxdt
