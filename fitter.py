import numpy as np
import casadi as ca
from scipy import optimize
from matplotlib import pyplot as plt

def argsplit(arg, n):
    try:
        assert len(arg)%n == 0
    except Exception as E:
        print(len(arg))
        raise E
    delims = [int(i*len(arg)/n) for i in range(n)] + [len(arg)]
    return [arg[delims[i]:delims[i+1]] for i in range(n)]

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

    def generate_objective(self, context, model):
        """Create the casadi objects that represent the inner objective function and its jacobian"""
        self.m = max(len(dataset['t']) for dataset in context.datasets)

        self.observation_vector = np.array(context.fitting_configuration['observation_vector'])
        self.observation_number = ca.MX.sym("m")
        self.weightings = np.array(context.fitting_configuration['weightings'][0])

        self.observations = [ca.MX.sym("y_"+str(i), self.m, 1)
                             for i in range(len(self.observation_vector))]
        self.collocation_matrix = ca.MX.sym("H", self.m, model.n)

        self.rho = ca.MX.sym("rho")
        self.default_rho = 10**(context.fitting_configuration['regularisation_parameter'][0])

        self.input_list = [model.ts, *model.cs, *model.ps, self.collocation_matrix,
                           self.observation_number, *self.observations, self.rho]

        self.create_inner_criterion(model)
        self.calculate_inner_jacobian(model)

    def create_inner_criterion(self, model):
        """Creates the inner objective function casadi object and function"""
        self._obj_1 = sum(self.weightings[i] * ca.norm_2(self.observations[i]
                                                         - self.collocation_matrix@model.xs[j])**2
                          for i, j in enumerate(self.observation_vector))/self.observation_number

        self._obj_fn1 = ca.Function("obj1", self.input_list, [self._obj_1])

        self._obj_2 = sum(ca.norm_2(model.get_xdash()[i] -
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


class Fitter():
    def __init__(self):
        self.objective_functions = []
        self.jacobian = None
        self.jacobian_function = None
        self.regularisation = None
        self.regularisation_derivative = None
        self.outer_objectives = []
        self.outer_jacobians = []
        self.problems = []
        self.solutions = dict()
        self.initial_guess = None
        self.initial_basis_coefs = None

        self._inner_objective = InnerObjective()

    def construct_objectives(self, context, model):
        self.initial_guess = context.initial_parameters
        self.initial_basis_coefs = 0.5 * np.ones(model.K * model.s)
        self._inner_objective.generate_objective(context, model)
        self.create_outer_jacobian(model)
        self.create_regularisation(context, model)
        for dataset in context.datasets:
            obj_fn, obj_jac = self._inner_objective.create_objective_functions(model, dataset)
            self.objective_functions.append(self.wrap(obj_fn, obj_jac))
            self.outer_objectives.append(self.outer_function(dataset, model))
            self.outer_jacobians.append(self.outer_jacobian_function(dataset, model))

    def create_regularisation(self, context, model):
        alpha = context.fitting_configuration['regularisation_parameter'][2]
        theta0 = context.fitting_configuration['regularisation_parameter'][3] # typically 1
        self.regularisation = lambda p: alpha * np.dot(p-theta0, p-theta0)
        self.regularisation_derivative = lambda p: 2*alpha*(p-theta0)

    def outer_function(self, dataset, model):
        def H(c, p, rho=None):
            if rho is None:
                rho = self._inner_objective.default_rho
            return (self._inner_objective._obj_fn1(model.observation_times, *argsplit(c, model.s),
                                                  *p, self._inner_objective.generate_collocation_matrix(dataset, model),
                                                  len(dataset['t']), *self._inner_objective.pad_observations(dataset['y']), rho)
                    + self.regularisation(p))
        return H

    def outer_jacobian_function(self, dataset, model):
        def J(c, p, rho=None):
            if rho is None:
                rho = self._inner_objective.default_rho
            return self.jacobian_function(model.observation_times, *argsplit(c, model.s), *p,
                                          self._inner_objective.generate_collocation_matrix(dataset, model),
                                          len(dataset['t']), *self._inner_objective.pad_observations(dataset['y']),
                                          rho, self.regularisation_derivative(p))
        return J

    @staticmethod
    def wrap(obj_fn, obj_jac):
        # create a function that solves the inner optimization problem
        def wrapd_fn(p, c0, rho=None):
            return optimize.minimize(obj_fn, c0, args=(p, rho), method="BFGS", jac=obj_jac)
        return wrapd_fn

    def create_outer_jacobian(self, model):
        dHdp = ca.MX.sym("outer_partial_p", 1, len(model.ps))
        dHdc = ca.hcat([ca.gradient(self._inner_objective._obj_1, ci) for ci in model.cs]).reshape((1, 3*model.K))
        d2Jdc2 = ca.hcat([ca.jacobian(self._inner_objective.inner_jacobian, ci) for ci in model.cs]).reshape((3*model.K, 3*model.K))
        d2Jdcdp = ca.hcat([ca.jacobian(self._inner_objective.inner_jacobian, pi) for pi in model.ps]).reshape((3*model.K, len(model.ps)))

        jacobian = dHdp - dHdc@ca.solve(d2Jdc2, d2Jdcdp)
        self.jacobian = jacobian
        self.jacobian_function = ca.Function('outer_jac',
                                             self._inner_objective.input_list + [dHdp],
                                             [jacobian])

    def construct_problems(self):
        for pars in zip(self.objective_functions, self.outer_objectives, self.outer_jacobians):
            new_problem = Problem(self.initial_guess, self.initial_basis_coefs,)
            new_problem.make(*pars)
            self.problems.append(new_problem)


    def solve(self, rho=None):
        if rho is None:
            rho = self._inner_objective.default_rho
        self.solutions[str(rho)] = []
        for problem in self.problems:
            self.solutions[str(rho)].append(problem.solve(rho))

    def visualise(self):
        plotting_array = []
        for rho in self.solutions:
            fun_sum = sum(float(solution.fun) for solution in self.solutions[rho])
            plotting_array.append((float(rho), fun_sum))
        plotting_array = np.array(plotting_array)
        plt.plot(*plotting_array.T, 'o')
        plt.show()

    def write(self, output_file):
        with open(output_file, "w") as ofh:
            ofh.write(self.solutions)

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
            key = "y".join(map(str, p)) + "r" + str(rho)
            sol = self.cache.get(key)
            if sol is None:
                sol = inn_solver(p, self.cache.recent, rho)
                self.cache.add(key, sol)
            return float(eval_fn(sol.x, p, rho))
        def j_evl(p, rho=None):
            key = "y".join(map(str, p)) + "r" + str(rho)
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
                                 })
