import copy
import numpy as np
import casadi as ca
from scipy import optimize
import modeller
import pickle

def argsplit(arg, n):
    try:
        assert len(arg)%n == 0
    except Exception as E:
        print(len(arg))
        raise E
    delims = [int(i*len(arg)/n) for i in range(n)] + [len(arg)]
    return [arg[delims[i]:delims[i+1]] for i in range(n)]

def tokey(root, branches):
    return f"{'y'.join(map(str, branches))}r{root}"

class InnerObjective():
    """Object that contains the ability to create the inner objective function"""
    def __init__(self):
        self.m = 0
        self.observations = None
        self.collocation_matrix = None
        self.observation_vector = None
        self.observation_number = 0
        self.weightings = None
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
        self.observation_number = ca.MX.sym("m")
        self.weightings = np.array(config['weightings'][0])

        self.observations = [ca.MX.sym("y_"+str(i), self.m, 1)
                             for i in range(len(self.observation_vector))]
        self.collocation_matrix = ca.MX.sym("H", self.m, model.n)

        self.rho = ca.MX.sym("rho")
        self.default_rho = 10**(config['regularisation_parameter'][0])

        self.input_list = [model.ts, *model.cs, *model.ps, self.collocation_matrix,
                           self.observation_number, *self.observations, self.rho]

        if self.inner_criterion is None:
            self.create_inner_criterion(model)
        if self.inner_jacobian is None:
            self.calculate_inner_jacobian(model)

    def create_inner_criterion(self, model):
        """Creates the inner objective function casadi object and function"""
        self._obj_1 = sum(self.weightings[i] * ca.norm_2(self.observations[i]
                                                         - self.collocation_matrix@model.xs[j])**2
                          for i, j in enumerate(self.observation_vector))/self.observation_number

        self._obj_fn1 = ca.Function("obj1", self.input_list, [self._obj_1])

        self._obj_2 = sum(ca.norm_fro(model.get_xdash()[:,i] -
                                    model.model(model.ts, *model.cs, *model.ps)[:, i])**2
                          for i in range(model.s))/model.n

        # nv1 = ca.hcat([ca.norm_fro(model.get_xdash()[i, :])  for i in range(model.n)])
        # nv2 = ca.hcat([ca.norm_fro(model.model(model.ts, *model.cs, *model.ps)[i, :])
        #                for i in range(model.n)])
        # xv1 = model.get_xdash()
        # xv2 = model.model(model.ts, *model.cs, *model.ps)
        # self._obj_2 = sum(2*ca.atan2(ca.norm_fro(nv2[i]@xv1[i, :] - nv1[i]@xv2[i, :]),
        #                              ca.norm_fro(nv2[i]@xv1[i, :] + nv1[i]@xv2[i, :]))
        #                   for i in range(model.n))/model.n

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
        colloc_matrix_numerical = np.zeros((self.m, model.n))
        for i, d_t in enumerate(dataset['t']):
            j = np.argmin(np.fabs(model.observation_times - d_t))
            colloc_matrix_numerical[i, j] = 1

        return colloc_matrix_numerical

    def pad_observations(self, observations):
        observations_shaped = np.vstack(observations).T

        assert len(observations_shaped) == len(self.observation_vector)

        padded_observations = np.zeros((len(self.observation_vector), self.m))
        for idx, observation in enumerate(observations_shaped):
            padded_observations[idx, 0:len(observation)] = observation

        return padded_observations

    def create_objective_functions(self, model, dataset):
        def obj_func(c, p, rho=None):
            if rho is None:
                rho = self.default_rho
            return float(self.inner_criterion_fn(model.observation_times, *argsplit(c, model.s),
                                                 *p,
                                                 self.generate_collocation_matrix(dataset, model),
                                                 len(dataset['t']),
                                                 *self.pad_observations(dataset['y']),
                                                 rho
                                                )
                        )

        def obj_jac(c, p, rho=None):
            if rho is None:
                rho = self.default_rho
            return np.array(self.inner_jacobian_fn(model.observation_times, *argsplit(c, model.s),
                                                   *p,
                                                   self.generate_collocation_matrix(dataset, model),
                                                   len(dataset['t']),
                                                   *self.pad_observations(dataset['y']),
                                                   rho
                                                  )
                           ).reshape(-1,)
        return obj_func, obj_jac

class Fitter():
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
        alpha = config['regularisation_parameter'][2]
        theta0 = config['regularisation_value']
        self.regularisation = lambda p: alpha * np.dot(p-theta0, p-theta0)
        self.regularisation_derivative = lambda p: 2*alpha*(p-theta0)

    def wrap_outer_objective(self, dataset, model, inner_objective):
        def H(c, p, rho=None):
            if rho is None:
                rho = inner_objective.default_rho
            return (inner_objective.inner_criterion_fn(model.observation_times, *argsplit(c, model.s),
                                             *p, inner_objective.generate_collocation_matrix(dataset, model),
                                             len(dataset['t']), *inner_objective.pad_observations(dataset['y']), rho)
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
        regularisation = ca.MX.sym("outer_partial_p", 1, len(model.ps))
        # dHdc = ca.hcat([ca.gradient(inner_objective._obj_1, ci) for ci in model.cs]).reshape((1, model.s*model.K))
        dJdp = ca.hcat([ca.gradient(inner_objective.inner_criterion, pi) for pi in model.ps]).reshape((1, len(model.ps)))
        jacobian = regularisation + dJdp
        return ca.Function('outer_jac',
                           inner_objective.input_list + [regularisation],
                           [jacobian])

    def wrap_outer_jacobian(self, dataset, model, inner_objective, jacobian_function):
        def dH(c, p, rho=None):
            if rho is None:
                rho = inner_objective.default_rho
            return jacobian_function(model.observation_times, *argsplit(c, model.s), *p,
                                     inner_objective.generate_collocation_matrix(dataset, model),
                                     len(dataset['t']), *inner_objective.pad_observations(dataset['y']),
                                     rho, self.regularisation_derivative(p))
        return dH

    def solve(self, rho=None):
        rhokey = str(rho)
        if str(rho) not in self.solutions.keys():
            self.solutions[rhokey] = []
        for problem in self.problems:
            self.solutions[rhokey].append(problem.solve(rho))

    def write(self, file="fitter.out"):
        with open(file, 'wb') as f:
            pickle.dump(file=f, obj=[self.solutions] + [p.cache.results for p in self.problems])

class CCache():
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
    def __init__(self, guess, c_0=None):
        self.function = None
        self.jacobian = None
        self.initial_guess = guess
        self.cache = CCache()
        # intialize the cache
        self.cache.recent = c_0
        self.bounds = optimize.Bounds(np.zeros(len(guess)), [np.inf]*len(guess))

    def make(self, inn_solver, eval_fn, jac_fn):
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
    def __init__(self, file=None):
        self.file = file
        self.solutions = None
        self.problem_cache = None
        if file:
            self.read()

    def read(self):
        with open(self.file, 'rb') as f:
            obj = pickle.load(f)
            self.solutions = obj[0]
            self.problem_cache = obj[1:]
