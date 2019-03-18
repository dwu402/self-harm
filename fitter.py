import numpy as np
import casadi as ca
from scipy import optimize


def argsplit(arg, n):
    try:
        assert len(arg)%n == 0
    except Exception as E:
        print(len(arg))
        raise E
    delims = [int(i*len(arg)/n) for i in range(n)] + [len(arg)]
    return [arg[delims[i]:delims[i+1]] for i in range(n)]

class InnerObjective():
    def __init__(self):
        self.m = 0
        self.observations = None
        self.collocation_matrix = None
        self.observation_vector = None
        self.observation_number = 0
        self.rho = 0
        self.input_list = None
        self.inner_criterion = None
        self.inner_jacobian = None
        self.inner_criterion_fn = None
        self.inner_jacobian_fn = None

        self._obj_1 = None
        self._obj_2 = None

    def generate_objective(self, context, model):

        self.m = int(model['observation_times'][-1])

        self.observation_vector = np.array(context['observation_vector'])
        self.observation_number = ca.MX.sym("m")

        self.observations = [ca.MX.sym("y_"+str(i), self.m, 1)
                             for i in range(len(self.observation_vector))]
        self.collocation_matrix = ca.MX.sym("H", self.m, model.n)

        self.input_list = [model.ts, *model.cs, *model.ps, self.collocation_matrix,
                           self.observation_number, *self.observations]

        self.rho = ca.MX.sym("rho")
        self.create_inner_criterion(model)
        self.calculate_inner_jacobian(model)

    def create_inner_criterion(self, model):
        self._obj_1 = sum(ca.norm_2(self.observations[i] - self.collocation_matrix@model.xs[j])**2
                          for i, j in enumerate(self.observation_vector))/self.observation_number

        self._obj_2 = sum(ca.norm_2(model.get_xdash[i] -
                                    model.model(model.ts, *model.cs, *model.ps)[:, i])**2
                          for i in range(model.s))/model.n

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
        assert len(observations) == len(self.observation_vector)

        padded_observations = np.zeros((len(self.observation_vector), self.m))
        for idx, observation in enumerate(observations):
            padded_observations[idx, :] = observation

        return padded_observations

    def create_objective_functions(self, model, dataset):
        def obj_func(c, p):
            return float(self.inner_criterion_fn(model.observation_times, *argsplit(c, model.s),
                                                 *p,
                                                 self.generate_collocation_matrix(dataset, model),
                                                 len(dataset['t']),
                                                 *argsplit(self.pad_observations(dataset['y']),
                                                           model.s)
                                                )
                        )

        def obj_jac(c, p):
            return np.array(self.inner_jacobian_fn(model.observation_times, *argsplit(c, model.s),
                                                   *p,
                                                   self.generate_collocation_matrix(dataset, model),
                                                   len(dataset['t']),
                                                   *argsplit(self.pad_observations(dataset['y']),
                                                             model.s)
                                                  )
                           ).reshape(-1,)
        return obj_func, obj_jac


class CCache():
    def __init__(self):
        self.recent = []
        self.results = dict()

    def add(self, key, value):
        self.results[key] = value
        self.recent = value

    def get(self, key):
        if key not in self.results.keys():
            return None
        else:
            return self.results.pop(key)


class Fitter():
    def __init__(self):
        self.objectives = []
        self.objective_functions = []
        self.cache = CCache()
        self.jacobians = []
        self.jacobian_functions = []

    def construct_objectives(self, context, model):
        for dataset in context['datasets']:
            new_objective = InnerObjective()
            new_objective.generate_objective(context, model)
            obj_fn, obj_jac = new_objective.create_objective_functions(model, dataset)

            self.objectives.append(new_objective)
            self.objective_functions.append(self.wrap(obj_fn, obj_jac))
            self.create_outer_jacobian(model, new_objective)

    """TODO: write this to be the outer objective function"""
    @staticmethod
    def wrap(obj_fn, obj_jac):
        # create a function that solves the inner optimization problem
        def wrapd_fn(p, c0):
            return optimize.minimize(obj_fn, c0, args=p, method="BFGS", jac=obj_jac)
        return wrapd_fn

    def create_outer_jacobian(self, model, objective):
        dHdp = ca.MX.sym("outer_partial_p", 1, len(model.ps))
        dHdc = ca.hcat([ca.gradient(objective._obj_1, ci) for ci in model.cs]).reshape((1, 3*model.K))
        d2Jdc2 = ca.hcat([ca.jacobian(objective.inner_jacobian, ci) for ci in model.cs]).reshape((3*model.K, 3*model.K))
        dJ2dcdp = ca.hcat([ca.jacobian(objective.inner_jacobian, pi) for pi in model.ps]).reshape((3*model.K, len(model.ps)))

        jacobian = dHdp - dHdc@ca.solve(d2Jdc2, dJ2dcdp)
        self.jacobians.append(jacobian)
        self.jacobian_functions.append(ca.Function('outer_jac',
                                                   objective.input_list + [dHdp],
                                                   [jacobian]))

    def prep(self, context, model):
        return
