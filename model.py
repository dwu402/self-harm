from pathlib import Path
from importlib import util as importutil
import scipy.integrate as spi


def get_model(model_file):
    """ get_model returns the model function defined in an arbitrary file """
    model_name = Path(model_file).stem
    try:
        model_spec = importutil.spec_from_file_location(model_name, model_file)
        if model_spec is None:
            raise ImportError
        model_module = importutil.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
    except ImportError:
        print(model_file, "not found")
        raise ImportError

    return model_module.model


def integrate_model(model_function, model_context):
    """ integrate_model integrates a target ODE according to the context """
    initial_values = model_context['initial_values']
    time_span = model_context['time_span']
    parameters = model_context['parameters']

    return spi.odeint(model_function, initial_values, time_span, args=(parameters,))
