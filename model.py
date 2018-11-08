from pathlib import Path
from importlib import util as importutil
import numpy as np
import scipy.integrate as spi
import ast


def import_module_from_file(file_name):
    """helper function to import an arbitrary file as module"""
    module_name = Path(file_name).stem
    try:
        module_spec = importutil.spec_from_file_location(module_name, file_name)
        if module_spec is None:
            raise ImportError
        module_module = importutil.module_from_spec(module_spec)
        module_spec.loader.exec_module(module_module)
    except ImportError:
        print(file_name, "not found")
        raise ImportError

    return module_module


def get_context(context_file):
    """Return the running context for the model"""
    if not Path(context_file).exists():
        raise FileNotFoundError
    with open(context_file) as cf:
        file_contents = cf.read().splitlines()

    context = {
        "time_span": [],
        "initial_values": [],
        "parameters": []
    }

    context['time_span'] = [float(value) for value in file_contents[0].split()]
    context['initial_values'] = [float(value) for value in file_contents[1].split()]
    for line in file_contents[2:]:
        parsed_line = line.split()
        ptype = parsed_line[0]
        pval = ' '.join(parsed_line[1:])
        if ptype in ('s', 'string'):
            context['parameters'].append(pval)
        elif ptype in ('f', 'float', 'i', 'int'):
            context['parameters'].append(float(pval))
        elif ptype in ('l', 'list'):
            context['parameters'].append(ast.literal_eval(pval))
        elif ptype in ('fn', 'function'):
            fn_file, fn_name = pval.split()
            fn_module = import_module_from_file(fn_file)
            context['parameters'].append(getattr(fn_module, fn_name))

    return context


def get_model(model_file):
    """Return the model function defined in an arbitrary file"""
    model_module = import_module_from_file(model_file)
    return model_module.model


def integrate_model(model_function, model_context):
    """Integrates a target ODE according to the context"""
    initial_values = model_context['initial_values']
    time_span = model_context['time_span']
    parameters = model_context['parameters']

    integrator = spi.ode(model_function).set_integrator('dopri5', nsteps=10**3)
    integrator.set_f_params(parameters).set_initial_value(initial_values, time_span[0])

    steps = int(time_span[2])
    dt = (time_span[1] - time_span[0]) / time_span[2]

    results = {
        't': [time_span[0]],
        'y': [initial_values]
    }
    step = 0
    while integrator.successful() and step < steps:
        step += 1
        result = integrator.integrate(integrator.t + dt)
        results['t'].append(integrator.t)
        results['y'].append(result)

    return results
