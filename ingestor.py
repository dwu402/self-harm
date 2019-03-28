import casadi as ca
from pathlib import Path
from importlib import util as importutil
import warnings
import pandas as pd


def check_file(file_name):
    """Checks to see if a file exists"""
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

def fn_from_file(file, function_name):
    """Helper function to get a function from a py file"""
    return getattr(import_module_from_file(file), function_name)

def initialise_context():
    """Initialises the context dictionary"""
    return {
        'context_files': {
            'model_file': '',
            'configuration_file': '',
            'data_files': []
        },
        'model': None,
        'initial_parameters': [],
        'initial_values': [],
        'time_span': [],
        'datasets': [],
        'data_parser': None,
        'modelling_configuration':{
            'grid_size': 0,
            'basis_number': 0,
            'model_form': None,
            'knot_function': None,
        },
        'fitting_configuration':{
            'refits': 0,
            'regularisation_parameter': [],
            'resampling_parameter': 0,
            'error_function': None,
        },
        'visualisation_function': None
    }

def read_run_file(context, run_file_name):
    """Parses the run file"""
    check_file(run_file_name)
    with open(run_file_name, 'r') as run_file:
        run_configs = run_file.read().splitlines()
    for config in run_configs:
        config_values = config.split()
        config_type = config_values.pop(0)
        if config_type == "#":
            continue
        elif config_type in ['mf', 'model-file']:
            context['context_files']['model_file'] = str(config_values[0])
        elif config_type in ['cf', 'configuration-file']:
            context['context_files']['configuration_file'] = str(config_values[0])
        elif config_type in ['df', 'data-file']:
            context['context_files']['data_files'].extend(config_values)
        else:
            error_string = "Unhandled config type: " + str(config_type)
            raise TypeError(error_string)

    parse_model_file(context)
    parse_config_file(context)
    parse_data_files(context)

def parse_model_file(context):
    """Reads the model file to return the function that specifies the model"""
    if not context['context_files']['model_file']:
        raise RuntimeError('No model file specified')
    else:
        check_file(context['context_files']['model_file'])

    model_module = import_module_from_file(context['context_files']['model_file'])
    context['model'] = model_module.model
    context['modelling_configuration']['model_form'] = model_module.model_form()

def parse_config_file(context):
    """Parses the configuration file"""
    if not context['context_files']['configuration_file']:
        warnings.warn('No configuration file specified', RuntimeWarning)
        return

    check_file(context['context_files']['configuration_file'])
    with open(context['context_files']['configuration_file']) as config_file:
        configs = config_file.read().splitlines()

    for config in configs:
        config_values = config.split()
        config_type = config_values[0]
        config_values = config_values[1:]
        if config_type == '#':
            continue
        elif config_type in ['iv', 'initial-values']:
            context['initial_values'] = [float(val) for val in config_values]
        elif config_type in ['ip', 'initial-parameters']:
            context['initial_parameters'] = [float(val) for val in config_values]
        elif config_type in ['ts', 'time-span']:
            context['time_span'] = [float(val) for val in config_values]
        elif config_type in ['dp', 'data-parser']:
            context['data_parser'] = fn_from_file(config_values[0], config_values[1])
        elif config_type in ['gs', 'grid-size']:
            context['modelling_configuration']['grid_size'] = int(config_values[0])
        elif config_type in ['bn', 'basis-number']:
            context['modelling_configuration']['basis_number'] = int(config_values[0])
        elif config_type in ['kf', 'knot-function']:
            context['modelling_configuration']['knot_function'] = fn_from_file(config_values[0], config_values[1])
        elif config_type in ['rf', 'refits']:
            context['fitting_configuration']['refits'] = float(config_values[0])
        elif config_type in ['rg', 'regularisation']:
            context['fitting_configuration']['regularisation_parameter'] = [float(val) for val in config_values]
        elif config_type in ['rs', 'resample']:
            context['fitting_configuration']['resampling_parameter'] = int(config_values[0])
        elif config_type in ['vf', 'visualisation-function']:
            context['visualisation_function'] = fn_from_file(config_values[0], config_values[1])
        else:
            error_string = "Unhandled config type: " + str(config_type)
            raise TypeError(error_string)

def parse_data_files(context):
    """Parses the data files"""
    if not context['context_files']['data_files']:
        warnings.warn('Data files not specified', RuntimeWarning)
        return
    all_raw_data = []
    for data_file in context['context_files']['data_files']:
        check_file(data_file)
        df_extension = data_file.split('.')[-1].lower()
        if df_extension == 'xlsx':
            all_raw_data.append(pd.read_excel(data_file))
        elif df_extension == "csv":
            all_raw_data.append(pd.read_csv(data_file))
        else:
            error_string = "Filetype not supported: " + str(df_extension)
            raise TypeError(error_string)
    clean_data, context_updates = context['data_parser'](context, all_raw_data)
    context['datasets'] = clean_data
    context.update(context_updates)
