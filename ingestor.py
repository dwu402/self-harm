from pathlib import Path
from importlib import util as importutil
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
    with open(context_file) as cf_handle:
        file_contents = cf_handle.read().splitlines()

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
