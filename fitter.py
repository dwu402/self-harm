import copy
import numpy as np
import casadi as ca
from scipy import optimize
import modeller
import pickle
from functions.clustering import find_paried_distances as fpd

def argsplit(arg, n):
    """ Used for splitting the values of c into 3 c vectors for the model """
    try:
        assert len(arg)%n == 0
    except Exception as E:
        print(len(arg))
        raise E
    delims = [int(i*len(arg)/n) for i in range(n)] + [len(arg)]
    return [arg[delims[i]:delims[i+1]] for i in range(n)]

def tokey(root, branches):
    """ rho/p hasher """
    return f"{'y'.join(map(str, branches))}r{root}"

class InnerObjective():
    """Object that contains the ability to create the inner objective function

    This represents:
    J(c|theta) = ||w*p*(y-H*Phi*c)||^2 + lambda*||D(Phi*c) - f(Phi*c, theta)||^2
    where:
      w = weightings on state (diagonal)
      p = data density (diagonal)
      y = stacked data vector
      H = collocation matrix/observation model
      Phi = spline basis
      c = spline coefficients
      D = differential operator
      f = process model
      theta = process parameters
    """
    def __init__(self):
        self.m = 0
        self.observations = None
        self.collocation_matrices = None
        self.observation_vector = None
        self.weightings = None
        self.densities = None
        self.rho = None
        self.default_rho = 0
        self.input_list = None
        self.inner_criterion = None
        self.inner_jacobian = None
        self.inner_criterion_fn = None
        self.inner_jacobian_fn = None

        self._obj_1 = None
        self._obj_fn1 = None
        self._obj_2 = None
        self._obj_fn2 = None

    def generate_objective(self, config, dataset, model):
        """Create the casadi objects that represent the inner objective function and its jacobian"""
        self.m = len(dataset['t'])

        self.observation_vector = np.array(config['observation_vector'])
        self.weightings = np.array(config['weightings'][0])

        y_as_np = np.stack(dataset['y'].to_numpy())
        self.densities = fpd(y_as_np[:,0], y_as_np[:,1])

        self.observations = [ca.MX.sym("y_"+str(i), self.m, 1)
                             for i in range(len(self.observation_vector))]
        self.collocation_matrices = ca.MX.sym("H", self.m, model.n, len(self.observation_vector))

        self.rho = ca.MX.sym("rho")
        self.default_rho = 10**(config['regularisation_parameter'][0])

        self.input_list = [model.ts, *model.cs, *model.ps, *self.collocation_matrices,
                           *self.observations, self.rho]

        if self.inner_criterion is None:
            self.create_inner_criterion(model)
        if self.inner_jacobian is None:
            self.calculate_inner_jacobian(model)

    def create_inner_criterion(self, model):
        """Creates the inner objective function casadi object and function"""
        self._obj_1 = sum(w * ca.norm_2(self.densities*(ov - (cm@model.xs[j])))**2
                          for j, ov, w, cm in zip(self.observation_vector,
                                                  self.observations,
                                                  self.weightings,
                                                  self.collocation_matrices))

        self._obj_fn1 = ca.Function("obj1", self.input_list, [self._obj_1])

        self._obj_2 = sum(ca.norm_fro(model.get_xdash()[:, i] -
                                      model.model(model.ts, *model.cs, *model.ps)[:, i])**2
                          for i in range(model.s))/model.n

        self._obj_fn2 = ca.Function("obj2", self.input_list, [self._obj_2])

        self.inner_criterion = self._obj_1 + self.rho*self._obj_2
        self.inner_criterion_fn = ca.Function('obj',
                                              self.input_list,
                                              [self.inner_criterion])

    def calculate_inner_jacobian(self, model):
        self.inner_jacobian = ca.vcat([ca.gradient(self.inner_criterion, ci) for ci in model.cs])
        self.inner_jacobian_fn = ca.Function('obj_jac',
                                             self.input_list,
                                             [self.inner_jacobian])


    def generate_collocation_matrix(self, dataset, model):
        """ Generate the matrix that represents the observation model, g

        This is a matrix, where the time points are mapped onto the finer time grid"""
        observation_counts = self.count_observations(dataset['y'])
        colloc_matrix_numerical = [np.zeros((self.m, model.n)) for i in observation_counts]
        for k, count in enumerate(observation_counts):
            for i, d_t in enumerate(dataset['t']):
                if i < count:
                    j = np.argmin(np.fabs(model.observation_times - d_t))
                    colloc_matrix_numerical[k][i, j] = 1

        return colloc_matrix_numerical

    def true_pad_observations(self, observations):
        """ Pad observations with zeros """
        observations_shaped = np.vstack(observations).T

        assert len(observations_shaped) == len(self.observation_vector)

        padded_observations = np.zeros((len(self.observation_vector), self.m))
        for idx, observation in enumerate(observations_shaped):
            padded_observations[idx, 0:len(observation)] = observation

        return padded_observations

    def pad_observations(self, observations, convert=True):
        """Transposes pandas array to numpy array"""
        arr = np.stack(observations.to_numpy()).T
        for arr_row in arr:
            if len(arr_row) < self.m:
                arr = np.pad(arr, ((0, 0), (0, self.m-len(arr[0]))), 'constant', constant_values=0)
        if convert:
            arr = np.nan_to_num(arr, copy=True)
        return arr

    def count_observations(self, observations):
        nparray = self.pad_observations(observations, convert=False)
        return np.array([len(obs[np.isfinite(obs)]) for obs in nparray])

    def create_objective_functions(self, model, dataset):
        """ Return callable function objects that represent the objective and jacobian """
        def obj_func(c, p, rho=None):
            if rho is None:
                rho = self.default_rho
            return float(self.inner_criterion_fn(model.observation_times, *argsplit(c, model.s),
                                                 *p,
                                                 *self.generate_collocation_matrix(dataset, model),
                                                 *self.pad_observations(dataset['y']),
                                                 rho
                                                )
                        )

        def obj_jac(c, p, rho=None):
            if rho is None:
                rho = self.default_rho
            return np.array(self.inner_jacobian_fn(model.observation_times, *argsplit(c, model.s),
                                                   *p,
                                                   *self.generate_collocation_matrix(dataset, model),
                                                   *self.pad_observations(dataset['y']),
                                                   rho
                                                  )
                           ).reshape(-1,)
        return obj_func, obj_jac


class Fitter():
    """ Class that solves the outer objective problems """
    def __init__(self, context=None):
        self.models = []
        self.inner_objectives = []
        self.regularisation = None
        self.regularisation_derivative = None
        self.problems = []
        self.solutions = dict()

        self.__initial_guess = None
        self.__initial_basis_coefs = []
        self.__inner_evaluation_functions = []
        self.__outer_objectives = []
        self.__outer_jacobian_objects = []
        self.__outer_jacobians = []

        if context is not None:
            self.parse_context(context)
            self.construct_models(context)
            self.construct_objectives(context)
            self.construct_problems()

    def parse_context(self, context):
        """ Parse in any relevant, static information """
        self.__initial_guess = context.initial_parameters

    def construct_models(self, context):
        """Creates a model that encapsualtes the basis for each dataset"""
        for idx, dataset in enumerate(context.datasets):
            config = self.__encapsulate_model_config(idx, context, dataset)
            self.models.append(modeller.Model(config))

    def construct_objectives(self, context):
        """Creates the outer objective functions and jacobians for solving"""
        self.create_regularisation(context.fitting_configuration)
        for idx, (dataset, model) in enumerate(zip(context.datasets, self.models)):
            inner_objective = InnerObjective()
            self.inner_objectives.append(inner_objective)
            inner_objective.generate_objective(context.fitting_configuration, dataset, model)
            obj_fn, obj_jac = inner_objective.create_objective_functions(model, dataset)
            self.__inner_evaluation_functions.append(self.wrap(obj_fn, obj_jac))
            self.__outer_objectives.append(self.wrap_outer_objective(dataset, model, inner_objective))
            self.__outer_jacobian_objects.append(self.create_jacobian_object(model, inner_objective))
            self.__outer_jacobians.append(self.wrap_outer_jacobian(dataset, model, inner_objective,
                                                                   self.__outer_jacobian_objects[idx]))
            self.__initial_basis_coefs.append(0.5 * np.ones(model.K * model.s))

    def construct_problems(self):
        """Creates the objects that contain the problems to solve"""
        for pars in zip(self.__inner_evaluation_functions, self.__outer_objectives,
                        self.__outer_jacobians, self.__initial_basis_coefs):
            new_problem = Problem(self.__initial_guess, pars[-1])
            new_problem.make(*pars[:-1])
            self.problems.append(new_problem)

    @staticmethod
    def __encapsulate_model_config(idx, context, dataset):
        """Creates a configuration to pass to the model object"""
        config = copy.copy(context.modelling_configuration)
        config['model'] = context.model
        config['dataset'] = dataset
        config['time_span'] = context.time_span[idx]
        return config

    @staticmethod
    def wrap(obj_fn, obj_jac):
        """Creates a function that solves the inner optimization problem"""
        def wrapd_fn(p, c0, rho=None):
            return optimize.minimize(obj_fn, c0, args=(p, rho), method="BFGS", jac=obj_jac)
        return wrapd_fn

    def create_regularisation(self, config):
        """ this is the L-regularisation of the objective function

        alpha*||theta - theta_0||^2
        """
        alpha = config['regularisation_parameter'][2]
        theta0 = config['regularisation_value']
        self.regularisation = lambda p: alpha * np.dot(p-theta0, p-theta0)
        self.regularisation_derivative = lambda p: 2*alpha*(p-theta0)

    def wrap_outer_objective(self, dataset, model, inner_objective):
        """ Wraps the inner objective for c-optimisation, given p, rho """
        def H(c, p, rho=None):
            if rho is None:
                rho = inner_objective.default_rho
            return (inner_objective.inner_criterion_fn(
                        model.observation_times, *argsplit(c, model.s),
                        *p, *inner_objective.generate_collocation_matrix(dataset, model),
                        *inner_objective.pad_observations(dataset['y']), rho)
                    + self.regularisation(p))
        return H

    # def create_dcdp_object(self, dataset, model, inner_objective):
    #     d2Jdc2 = ca.hcat([ca.jacobian(inner_objective.inner_jacobian, ci) for ci in model.cs]).reshape((model.s*model.K, model.s*model.K))
    #     d2Jdcdp = ca.hcat([ca.jacobian(inner_objective.inner_jacobian, pi) for pi in model.ps]).reshape((model.s*model.K, len(model.ps)))
    #     dcdp = ca.solve(d2Jdc2, d2Jdcdp)
    #     dcdp_fn = ca.Function('dcdp', inner_objective.input_list, [dcdp])
    #     def dcdp_wrapper(c, p, rho=None):
    #         if rho is None:
    #             rho = inner_objective.default_rho
    #         return dcdp_fn(model.observation_times, *argsplit(c, model.s), *p,
    #                        inner_objective.generate_collocation_matrix(dataset, model),
    #                        len(dataset['t']), *inner_objective.pad_observations(dataset['y']), rho)
    #     return dcdp, dcdp_wrapper

    def create_jacobian_object(self, model, inner_objective):
        """ Create the jacobian object and function of the outer objective """
        regularisation = ca.MX.sym("outer_partial_p", 1, len(model.ps))
        # dHdc = ca.hcat([ca.gradient(inner_objective._obj_1, ci) for ci in model.cs]).reshape((1, model.s*model.K))
        dJdp = ca.hcat([ca.gradient(inner_objective.inner_criterion, pi) for pi in model.ps]).reshape((1, len(model.ps)))
        jacobian = regularisation + dJdp
        return ca.Function('outer_jac',
                           inner_objective.input_list + [regularisation],
                           [jacobian])

    def wrap_outer_jacobian(self, dataset, model, inner_objective, jacobian_function):
        """ Wraps the outer objective for p-optimisation, given the optimal c, and rho """
        def dH(c, p, rho=None):
            if rho is None:
                rho = inner_objective.default_rho
            return jacobian_function(model.observation_times, *argsplit(c, model.s), *p,
                                     *inner_objective.generate_collocation_matrix(dataset, model),
                                     *inner_objective.pad_observations(dataset['y']),
                                     rho, self.regularisation_derivative(p))
        return dH

    def solve(self, rho=None, propagate=False):
        rhokey = str(rho)
        if str(rho) not in self.solutions.keys():
            self.solutions[rhokey] = []
        for i, problem in enumerate(self.problems):
            self.solutions[rhokey].append(problem.solve(rho))
            if propagate:
                problem.initial_guess = self.solutions[rhokey][i].x

    def write(self, file="fitter.out"):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=[[[str(p) for p in m.ps] for m in self.models]] +
                        [self.solutions] +
                        [p.cache.results for p in self.problems])

    def read(self, file):
        reader = FitReader(file)
        self.solutions = reader.solutions
        for myp, rp in zip(self.problems, reader.problem_cache):
            myp.cache.results = rp

class CCache():
    """ Memoization helper object """
    def __init__(self):
        self.recent = []
        self.results = dict()

    def add(self, key, value):
        self.results[key] = value
        self.recent = value.x

    def get(self, key):
        if key not in self.results.keys():
            return None
        else:
            return self.results[key]

class Problem():
    """ Representation of an outer objective problem """
    def __init__(self, guess, c_0=None):
        self.function = None
        self.jacobian = None
        self.initial_guess = guess
        self.cache = CCache()
        # intialize the cache
        self.cache.recent = c_0
        self.bounds = optimize.Bounds(np.zeros(len(guess)), [np.inf]*len(guess))

    def make(self, inn_solver, eval_fn, jac_fn):
        """ Construct wrapper functions that solve the inner objective problem """
        def f_evl(p, rho=None):
            key = tokey(rho, p)
            sol = self.cache.get(key)
            if sol is None:
                sol = inn_solver(p, self.cache.recent, rho)
                self.cache.add(key, sol)
            return float(eval_fn(sol.x, p, rho))
        def j_evl(p, rho=None):
            key = tokey(rho, p)
            sol = self.cache.get(key)
            if sol is None:
                sol = inn_solver(p, self.cache.recent, rho)
                self.cache.add(key, sol)
            return np.array(jac_fn(sol.x, p, rho)).reshape(-1,)

        self.function = f_evl
        self.jacobian = j_evl

    def solve(self, rho):
        return optimize.minimize(self.function, self.initial_guess, args=rho,
                                 method="L-BFGS-B", jac=self.jacobian, bounds=self.bounds,
                                 options={
                                     "maxls": 1000,
                                     "gtol": 1e-06,
                                     "maxcor": 15,
                                 })


class FitReader():
    """ unpickler """
    def __init__(self, file=None):
        self.file = file
        self.ps = []
        self.solutions = None
        self.problem_cache = None
        if file:
            self.read()

    def read(self, old=False):
        with open(self.file, 'rb') as f:
            obj = pickle.load(f)
            if old or not isinstance(obj[0], list):
                self.solutions = obj[0]
                self.problem_cache = obj[1:]
            else:
                self.ps = obj[0]
                self.solutions = obj[1]
                self.problem_cache = obj[2:]
