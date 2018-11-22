from pathlib import Path
from importlib import util as importutil
import ast
import pandas as pd

def check_file(file_name):
    if not Path(file_name).exists():
        raise FileNotFoundError(file_name + " not found.")

def import_module_from_file(file_name):
    """helper function to import an arbitrary file as module"""
    check_file(file_name)
    module_name = Path(file_name).stem
    try:
        module_spec = importutil.spec_from_file_location(module_name, file_name)
        if module_spec is None:
            raise ImportError
        module_module = importutil.module_from_spec(module_spec)
        module_spec.loader.exec_module(module_module)
    except ImportError:
        error_string = file_name +  " not found"
        raise ImportError(error_string)

    return module_module


def initialise_context():
    """Intialises the context dictionary"""
    return {
        'model': lambda x: None,
        'parameters': [],
        'initial_values': None,
        'time_span': None,
        'data': None,
        'data_configuration': lambda x: None,
        'error_function': lambda x: None
    }

def get_parameters(context):
    """Parses the parameter file into the context"""
    parameter_file = context['parameter_file']
    check_file(parameter_file)
    with open(parameter_file) as pf_handle:
        file_contents = pf_handle.read().splitlines()

    for line in file_contents:
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
        else:
            error_string = "Unhandled parameter type: " + str(ptype)
            raise TypeError(error_string)

    return context


def get_model(context):
    """Return the model function defined in an arbitrary file"""
    model_file = context['model_file']
    check_file(model_file)
    model_module = import_module_from_file(model_file)
    context['model'] = model_module.model

def get_data(context):
    """Return the data for fitting from an arbitrary file"""
    data_file = context['data_file']
    check_file(data_file)
    df_extension = data_file.split('.')[-1].lower()
    if df_extension == 'xlsx':
        raw_data = pd.read_excel(data_file)
    elif df_extension == "csv":
        raw_data = pd.read_csv(data_file)
    else:
        error_string = "Filetype not supported: " + str(df_extension)
        raw_data = []
        raise TypeError(error_string)

    context['data'] = context['parse_data'](raw_data)

def fn_from_file(file, function_name):
    """Helper function to get a function from a py file"""
    return getattr(import_module_from_file(file), function_name)

def get_config(context, config_file):
    """Parses the configuration file into the context"""
    check_file(config_file)
    with open(config_file, 'r') as config_file_contents:
        configs = config_file_contents.read().splitlines()
        for config in configs:
            config_values = config.split()
            config_type = config_values.pop(0)
            if config_type in ['ts', 'time_span']:
                context['time_span'] = [float(value) for value in config_values]
            elif config_type in ['iv', 'initial_values']:
                context['initial_values'] = [float(value) for value in config_values]
            elif config_type in ['mf', 'model_file']:
                context['model_file'] = str(config_values[0])
            elif config_type in ['pf', 'parameter_file']:
                context['parameter_file'] = str(config_values[0])
            elif config_type in ['df', 'data_file']:
                context['data_file'] = str(config_values[0])
            elif config_type in ['pd', 'parse_data']:
                context['parse_data'] = fn_from_file(config_values[0], config_values[1])
            elif config_type in ['ef', 'error_function']:
                context['error_function'] = fn_from_file(config_values[0], config_values[1])
            else:
                error_string = "Unhandled config type: " + str(config_type)
                raise TypeError(error_string)
