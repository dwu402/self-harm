"""Module responsible for evaluating a model"""
import scipy.integrate as spi


def integrate_model(model_function, initial_values, time_span, parameters):
    """Integrates a target ODE according to the context"""
    integrator = spi.ode(model_function).set_integrator('dopri5', nsteps=1e3)
    integrator.set_f_params(parameters).set_initial_value(initial_values, time_span[0])

    steps = int(time_span[2])
    time_step = (time_span[1] - time_span[0]) / time_span[2]

    results = {
        't': [time_span[0]],
        'y': [initial_values]
    }
    step = 0
    while integrator.successful() and step < steps:
        step += 1
        result = integrator.integrate(integrator.t + time_step)
        results['t'].append(integrator.t)
        results['y'].append(result)

    return results
