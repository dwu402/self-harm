"""Module responsible for perfomring parameter inference/fitting"""
import numpy as np
from scipy import optimize as sciopt
import model


class FitterReturnCollection:
    """A class that defines the object that contains multiple FitterReturnObjects"""
    def __init__(self):
        self.results = []
        self.failures = []
        self.parameters = []

    def is_individual(self):
        return False

    def add_result(self, result):
        if result.is_success():
            self.results.append(result)
        else:
            self.failures.append(result)

    def build_parameters(self):
        self.parameters = np.array([r.get_parameters() for r in self.results])

    def get_mean_parameters(self):
        self.build_parameters()
        return np.mean(self.parameters, axis=0)

    def get_var_parameters(self):
        self.build_parameters()
        return np.var(self.parameters, axis=0)

    def get_parameters(self):
        self.build_parameters()
        return self.parameters

    def get_errors(self):
        error_list = [r.get_errors() for r in self.failures]
        return error_list

    def is_success(self):
        return not self.failures

    def get_statistics(self):
        if not self.results:
            return "No results found."

        statistics = []

        means = self.get_mean_parameters()
        variances = self.get_var_parameters()

        for parameter in range(len(means)):
            statistics.append(f"{means[parameter]} ~({np.sqrt(variances[parameter])})")

        return statistics


class FitterReturnObject:
    """A class that defines the object that is returned from fitting"""
    def __init__(self):
        self.error = None
        self.error_string = ""
        self.value = None
        self.parameters = []
        self.success_flag = False

    def is_individual(self):
        return True

    def is_success(self):
        return self.success_flag

    def get_errors(self):
        return self.error_string

    def get_value_string(self):
        return str(self.value)

    def get_parameters(self):
        return self.parameters

    def push_error(self, error):
        self.error = error
        self.error_string += str(error)
        self.error_string += "\n"

    def push_result(self, value, parameters):
        self.value = value
        self.parameters = np.array(np.abs(parameters))

    def push_failure(self, error, value, parameters):
        self.success_flag = False
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
    wrapped_fn = lambda p: context['error_function'](resample(context['data'], context['seed']),
                                                     fn_with_only_p(p),
                                                     p,
                                                     context['regularisation'])
    return wrapped_fn


def resample(data, seed):
    new_data = dict()
    for column in data:
        new_data[column] = np.array([data[column][idx] for idx in seed])
    return new_data


def generate_resampling_seed(context):
    resampling_parameter = 0
    total_length = len(context['data']['t'])
    n_points = total_length - resampling_parameter
    context['seed'] = sorted(np.random.choice(np.arange(total_length), size=n_points, replace=False))


def fitter(context):
    """Perform parameter estimation for a model"""
    return_obj = FitterReturnObject()

    # create the list of initial parameters
    # default to zero is TypeError
    p_0 = []
    for param in context['parameters']:
        try:
            float_of_p = float(param) * (0.5*np.random.rand()+0.75)
            p_0.append(float_of_p)
        except TypeError as _:
            p_0.append(0)

    res = None
    try:
        generate_resampling_seed(context)
        res = sciopt.minimize(wrap_function(context), p_0, method="Nelder-Mead", options={'disp':True, 'maxiter':1e5, 'maxfev':1e5})
        if not res.success:
            print(res)
            raise Exception("Fitting not successful")
        return_obj.push_success(res['fun'], res['x'])
    except Exception as exception:
        if res is None:
            res = dict()
        if 'fun' not in res.keys():
            res['fun'] = None
        if 'x' not in res.keys():
            res['x'] = []
        return_obj.push_failure(exception, res['fun'], res['x'])

    return return_obj
