import numpy as np
from matplotlib import pyplot as plt
from fitter import tokey, argsplit
from casadi import Function, hessian

class DataPlotter():
    def __init__(self, context):
        self.figure = None
        self.canvas = None
        self.view_function = context.visualisation_function
        self.data = context.datasets

    def __refresh__(self):
        self.figure = plt.figure()
        self.canvas = self.figure.add_subplot(111)

    def __call__(self):
        self.__refresh__()
        for data in self.data:
            self.view_function(self.canvas, data)
        self.figure.show()

class Plotter():
    def __init__(self, context=None, fitter=None):
        self.context = context
        self.fitter = fitter
        self.rhos = []
        self.sol_objs = []
        self.nState = 0
        if context and fitter:
            self.__parse_solution()

    def __parse_solution(self):
        self.rhos = [float(rho) for rho in self.fitter.solutions.keys()]
        self.sol_objs = self.fitter.solutions.values()
        self.nState = self.context.modelling_configuration['model_form']['state']

    def show_parameter_values(self, problem=0):
        """Shows the parameter values at each rho value for a given problem (data set)"""
        vals = [val[problem].x for val in self.sol_objs]
        plt.plot(self.rhos, vals, 'X-')
        plt.legend([str(p).replace('_', '') for p in self.fitter.models[problem].ps], loc="best",
                   bbox_to_anchor=(1.01, 1))
        plt.xscale('log')
        plt.yscale('log', nonposy="mask")
        plt.show()

    def show_iterations(self, problem=0, low=5):
        """Shows the number of iterations at each rho value for a given problem (data set)"""
        iters_list = np.array([[rho, val[problem].nit]
                               for rho, val in zip(self.rhos, self.sol_objs)])
        fevs_list = np.array([[rho, val[problem].nfev]
                              for rho, val in zip(self.rhos, self.sol_objs)])
        plt.semilogx(*iters_list.T, *fevs_list.T)
        low_iters = np.array([[k, v] for k,v in iters_list if v < low])
        plt.plot(*low_iters.T, 'ro')
        plt.legend(["niters", "nfev", "low iter"],
                   loc="best", bbox_to_anchor=(1.01, 1))
        plt.title("Algorithm evaluations")
        plt.xlabel(r"$\rho$")
        plt.ylabel("Number")
        plt.show()

    def p_of(self, target_rho, problem):
        """Helper method to get parameter values"""
        return self.fitter.solutions[str(target_rho)][problem].x

    def print_solution(self, target_rho, problem=0, label_vector=None):
        """Prints the parameter values"""
        if not label_vector:
            label_vector = [str(p).replace('_', '') for p in self.fitter.models[problem].ps]
        for param, pval in zip(label_vector, self.p_of(target_rho, problem)):
            print(f"par {param}={pval}")

    def draw_solution(self, target_rho, problem=0, plane='t', datakeys=None):
        """Plots the solution for a given projection plane"""
        getx = Function("getx", [self.fitter.models[problem].ts,
                                 *self.fitter.models[problem].cs],
                        self.fitter.models[problem].xs)
        problem_obj = self.fitter.problems[problem]
        c_end = problem_obj.cache.results[tokey(target_rho, self.p_of(target_rho, problem))].x
        times = self.fitter.models[problem].observation_times
        xs_end = np.array([np.array(i) for i in getx(times, *argsplit(c_end, self.nState))])
        data_values = self.context.datasets[problem]['y']
        if plane == 't':
            # time series plot
            nT = len(times)
            plt.plot(times, xs_end.reshape(self.nState, nT).T,
                     self.context.datasets[problem]['t'],
                     np.array([list(x) for x in data_values]), 'o')
            plt.legend([f"x {i}" for i in range(self.nState)] +
                       [f"data {i}" for i, _ in enumerate(data_values)],
                       loc="best", bbox_to_anchor=(1.01, 1))
            plt.xlabel('time, t')
            plt.show()
        elif len(plane) == 2:
            xaxis, yaxis = plane
            spline_dfield = np.array([self.context.model(t, xs_end[:, i],
                                                         self.p_of(target_rho, problem))
                                      for i, t in enumerate(times)])
            spline_dfield = spline_dfield.reshape(spline_dfield.shape[:2])
            plt.plot(xs_end[xaxis], xs_end[yaxis], 'o-')
            if datakeys:
                plt.plot(self.context.datasets[problem][datakeys[0]],
                         self.context.datasets[problem][datakeys[1]],
                         alpha=0.55)
            plt.quiver(xs_end[xaxis], xs_end[yaxis], spline_dfield[:, xaxis], spline_dfield[:, yaxis],
                       scale=None, angles='xy', headwidth=3, headlength=4.5, headaxislength=4,
                       width=0.0025)
            if datakeys:
                plt.xlabel(datakeys[0])
                plt.ylabel(datakeys[1])
                plt.legend(['Spline', 'Data', 'Gradients'])
            else:
                plt.xlabel(f"state variable {xaxis}")
                plt.ylabel(f"state variable {yaxis}")
                plt.legend(['Spline', 'Gradients'])
            plt.show()
        else:
            raise AssertionError("plane argument is incorrect")

    def draw_lcurve(self, problem=0):
        """Plots the Lcurve for the inner objective function"""
        times = self.fitter.models[problem].observation_times
        datafit_fn = lambda r, v: self.fitter.inner_objectives[problem]._obj_fn1(
            times,
            *argsplit(self.fitter.problems[problem].cache.results[tokey(r, v[problem].x)].x,
                      self.nState),
            *v[problem].x,
            self.fitter.inner_objectives[problem].generate_collocation_matrix(
                self.context.datasets[problem], self.fitter.models[problem]
            ),
            len(self.context.datasets[problem]['t']),
            *self.fitter.inner_objectives[problem].pad_observations(
                self.context.datasets[problem]['y']
            ),
            r
        )
        dfield_fn = lambda r, v: self.fitter.inner_objectives[problem]._obj_fn2(
            times,
            *argsplit(self.fitter.problems[problem].cache.results[tokey(r, v[problem].x)].x,
                      self.nState),
            *v[problem].x,
            self.fitter.inner_objectives[problem].generate_collocation_matrix(
                self.context.datasets[problem], self.fitter.models[problem]
            ),
            len(self.context.datasets[problem]['t']),
            *self.fitter.inner_objectives[problem].pad_observations(
                self.context.datasets[problem]['y']
            ),
            r
        )
        datafit_values = np.array([[r, datafit_fn(r, v)] for r, v in zip(self.rhos, self.sol_objs)])
        dfield_values = np.array([[r, dfield_fn(r, v)] for r, v in zip(self.rhos, self.sol_objs)])

        plt.loglog(datafit_values[:, 1], dfield_values[:, 1], '--o', linewidth=0.25)
        plt.xlabel("Data Fit")
        plt.ylabel("Diff Field")
        plt.show()

    def draw_confidence(self, target_rho, problem=0, label_vector=None):
        """Plots the confidence interval estiamtes based on 'Fisher information' (curvature)"""
        ps_end = self.p_of(target_rho, problem)
        fisher = []
        for p in self.fitter.models[problem].ps:
            H, g = hessian(self.fitter.inner_objectives[problem].inner_criterion, p)
            hfn = Function('hfn', self.fitter.inner_objectives[problem].input_list, [H])
            fisher.append(float(hfn(
                self.fitter.models[problem].observation_times,
                *argsplit(self.fitter.problems[problem].cache.results[tokey(target_rho, ps_end)].x,
                          self.nState),
                *ps_end,
                self.fitter.inner_objectives[problem].generate_collocation_matrix(
                    self.context.datasets[problem], self.fitter.models[problem]
                ),
                len(self.context.datasets[problem]['t']),
                *self.fitter.inner_objectives[problem].pad_observations(
                    self.context.datasets[problem]['y']
                ),
                target_rho
            )))
        pidx = range(len(ps_end))
        plt.bar(pidx, ps_end)
        plt.errorbar(pidx, ps_end, yerr=3*np.sqrt(1/np.array(fisher)), capsize=7, markeredgewidth=2,
                     linestyle='None', ecolor='k', color='k')
        if label_vector:
            plt.xticks(pidx, label_vector)
        else:
            plt.xticks(pidx, [str(p).replace('_', '') for p in self.fitter.models[problem].ps])
        plt.title("Confidence Interval Estimates using Fisher Information")
        plt.show()

    def draw_error(self, indexkeys, target_rho, problem=0):
        """Plots a representation of how much data fit error there is"""
        getx = Function("getx", [self.fitter.models[problem].ts,
                                 *self.fitter.models[problem].cs],
                        self.fitter.models[problem].xs)
        problem_obj = self.fitter.problems[problem]
        c_end = problem_obj.cache.results[tokey(target_rho, self.p_of(target_rho, problem))].x
        times = self.fitter.models[problem].observation_times
        xs_end = np.array([np.array(i) for i in getx(times, *argsplit(c_end, self.nState))])
        collocator = self.fitter.inner_objectives[problem].generate_collocation_matrix(
            self.context.datasets[problem], self.fitter.models[problem]
        )
        observables = [collocator@xs_end[i] for i in indexkeys.keys()]
        errs = [np.abs(observable.T - np.array(self.context.datasets[problem][datakey])).reshape(-1,)
                for observable, datakey in zip(observables, indexkeys.values())]
        if len(indexkeys) == 2:
            i0s = xs_end[list(indexkeys.keys()), :]
            scales = [np.mean(err) + 1.96*np.std(err) for err in errs]
            circles = [i0 + iscl*fn(np.linspace(0, 2*np.pi, 50))
                       for i0, iscl, fn in zip(i0s, scales, [np.cos, np.sin])]
            _, axes = plt.subplots(nrows=2, ncols=1)
            axis = axes[0]
            axes[1].plot(*i0s)
            axes[1].plot(*[self.context.datasets[problem][datakey]
                           for datakey in indexkeys.values()], 'o--', alpha=0.55)
            axes[1].plot(circles[0], circles[1], 'ko', alpha=0.2)
        else:
            axis = plt.subplot(111)
        for err in errs:
            axis.plot(self.context.datasets[problem]['t'], err)
        plt.show()
