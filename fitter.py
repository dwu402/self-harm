import numpy as np
from scipy import optimize as sciopt
import model


class FitterReturnObject:
    """A class that defines the object that is returned from fitting"""
    def __init__(self):
        self.error_string = ""
        self.value_string = ""
        self.parameter_string = ""
        self.success_flag = False

    def is_success(self):
        return self.success_flag

    def get_errors(self):
        return self.error_string

    def get_value(self):
        return self.value_string

    def get_parameters(self):
        return self.parameter_string

    def push_error(self, error):
        self.success_flag = False
        self.error_string += error
        self.error_string += "\n"

    def push_result(self, value, parameters):
        self.value_string = str(value)
        self.parameter_string = ""
        for parameter in parameters:
            pstring = "".join((parameter['name'], ": ", str(parameter['value']), "\n"))
            self.parameter_string += pstring

    def push_failure(self, error, value, parameters):
        self.push_error(error)
        self.push_result(value, parameters)

    def push_success(self, value, parameters):
        self.success_flag = True
        self.push_result(value, parameters)


def wrap_function(context):
    """Wraps a function so that the only input arguments are the parameters"""
    fn_with_only_p = lambda p: model.integrate_model(context['model'],
                                                     context['initial_values'],
                                                     context['time_span'],
                                                     p)
    wrapped_fn = lambda p: context['error_function'](fn_with_only_p(p))
    return wrapped_fn


def fitter(context):
    """Perform parameter estimation for a model"""
    return_obj = FitterReturnObject()

    # create the list of initial parameters
    # default to zero is TypeError
    p_0 = []
    for param in context['parameters']:
        try:
            float_of_p = float(param)
            p_0.append(float_of_p)
        except TypeError as _:
            p_0.append(0)

    try:
        res = sciopt.minimize(wrap_function(context), p_0,
                        method="nelder-mead", options={'disp':True})
        if not res.success:
            raise Exception("Fitting not successful")
        return_obj.push_success(res['fun'], res['x'])
    except Exception as exception:
        if 'fun' not in res.keys():
            res['fun'] = None
        if 'x' not in res.keys():
            res['x'] = None
        return_obj.push_failure(exception, res['fun'], res['x'])

    return return_obj
