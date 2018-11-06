import numpy as np
import scipy.integrate as spi


def get_model(model_file):
    """ get_model returns the model function defined in an arbitrary file """
    return __import__(model_file).model


def integrate_model(model_function, model_context):
    """ integrate_model integrates a target ODE according to the context """
    initial_values = model_context['initial_values']
    time_span = model_context['time_span']
    parameters = model_context['parameters']

    return spi.odeint(model_function, initial_values, time_span, args=(parameters,))
