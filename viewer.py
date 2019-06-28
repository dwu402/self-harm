import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache
from fitter import tokey, argsplit
from casadi import Function, hessian

def setup_canvas(size=False,ipy=False):
    """Setups up the style and colours available when plotting"""
    # style settings
    plt.style.use('seaborn-notebook')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    if size is True:
        plt.rcParams['figure.figsize'] = [15, 10]
    elif size:
        plt.rcParams['figure.figsize'] = size
    if ipy:
        plt.rcParams['figure.dpi'] = 72
        plt.rcParams['figure.subplot.bottom'] = 0.125
        plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
        plt.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
    # include more colours
    from cycler import cycler
    new_colours = cycler(color=["k", "m"])
    plt.rcParams['axes.prop_cycle'] = plt.rcParams['axes.prop_cycle'].concat(new_colours)

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

    def new_figure(self, with_axes=None):
        if isinstance(with_axes, (list, tuple)):
            fig, axes = plt.subplots(*with_axes)
        elif isinstance(with_axes, dict):
            fig, axes = plt.subplots(**with_axes)
        else:
            fig, axes = plt.subplots()
        return fig, axes

    def show_parameter_values(self, problem=0, labels=None):
        """Shows the parameter values at each rho value for a given problem (data set)"""
        vals = [val[problem].x for val in self.sol_objs]
        fig, ax = self.new_figure()
        ax.plot(self.rhos, vals, 'X-')
        if not labels:
            labels = [str(p).replace('_', '') for p in self.fitter.models[problem].ps]
        ax.legend(labels, loc="best", bbox_to_anchor=(1.01, 1))
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        plt.show()

    def show_iterations(self, problem=0, low=5):
        """Shows the number of iterations at each rho value for a given problem (data set)"""
        iters_list = np.array([[rho, val[problem].nit]
                               for rho, val in zip(self.rhos, self.sol_objs)])
        fevs_list = np.array([[rho, val[problem].nfev]
                              for rho, val in zip(self.rhos, self.sol_objs)])
        fig, ax = self.new_figure()
        ax.semilogx(*iters_list.T, *fevs_list.T)
        low_iters = np.array([[k, v] for k, v in iters_list if v < low])
        ax.plot(*low_iters.T, 'ro')
        ax.legend(["niters", "nfev", "low iter"],
                  loc="best", bbox_to_anchor=(1.01, 1))
        ax.set_title("Algorithm evaluations")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("Number")
        plt.show()

    def p_of(self, target_rho, problem):
        """Helper method to get parameter values"""
        return self.fitter.solutions[str(target_rho)][problem].x

    def print_solution(self, target_rho, problem=0, labels=None):
        """Prints the parameter values"""
        if not labels:
            labels = [str(p).replace('_', '') for p in self.fitter.models[problem].ps]
        for param, pval in zip(labels, self.p_of(target_rho, problem)):
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
        fig, ax = self.new_figure()
        if plane == 't':
            # time series plot
            nT = len(times)
            ax.plot(times, xs_end.reshape(self.nState, nT).T,
                    self.context.datasets[problem]['t'],
                    np.array([list(x) for x in data_values]), 'o')
            ax.legend([f"x {i}" for i in range(self.nState)] +
                      [f"data {i}" for i, _ in enumerate(data_values)],
                      loc="best", bbox_to_anchor=(1.01, 1))
            ax.set_xlabel('time, t')
            plt.show()
        elif len(plane) == 2:
            xaxis, yaxis = plane
            spline_dfield = np.array([self.context.model(t, xs_end[:, i],
                                                         self.p_of(target_rho, problem))
                                      for i, t in enumerate(times)])
            spline_dfield = spline_dfield.reshape(spline_dfield.shape[:2])
            ax.plot(xs_end[xaxis], xs_end[yaxis], 'o-')
            if datakeys:
                ax.plot(self.context.datasets[problem][datakeys[0]],
                        self.context.datasets[problem][datakeys[1]],
                        'o-', alpha=0.55)
            ax.quiver(xs_end[xaxis], xs_end[yaxis], spline_dfield[:, xaxis], spline_dfield[:, yaxis],
                      scale=None, angles='xy', headwidth=3, headlength=4.5, headaxislength=4,
                      width=0.0025)
            if datakeys:
                ax.set_xlabel(datakeys[0])
                ax.set_ylabel(datakeys[1])
                ax.legend(['Spline', 'Data', 'Gradients'], loc="best", bbox_to_anchor=(1.01, 1))
            else:
                ax.set_xlabel(f"state variable {xaxis}")
                ax.set_ylabel(f"state variable {yaxis}")
                ax.legend(['Spline', 'Gradients'], loc="best", bbox_to_anchor=(1.01, 1))
            plt.show()
        else:
            raise AssertionError("plane argument is incorrect")

    def draw_lcurve(self, problem=0, target_rho=None, optimal=False):
        """Plots the Lcurve for the inner objective function"""
        times = self.fitter.models[problem].observation_times
        datafit_fn = lambda r, v: self.fitter.inner_objectives[problem]._obj_fn1(
            times,
            *argsplit(self.fitter.problems[problem].cache.results[tokey(r, v[problem].x)].x,
                      self.nState),
            *v[problem].x,
            *self.fitter.inner_objectives[problem].generate_collocation_matrix(
                self.context.datasets[problem], self.fitter.models[problem]
            ),
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
            *self.fitter.inner_objectives[problem].generate_collocation_matrix(
                self.context.datasets[problem], self.fitter.models[problem]
            ),
            *self.fitter.inner_objectives[problem].pad_observations(
                self.context.datasets[problem]['y']
            ),
            r
        )
        datafit_values = np.array([[r, datafit_fn(r, v)] for r, v in zip(self.rhos, self.sol_objs)])
        dfield_values = np.array([[r, dfield_fn(r, v)] for r, v in zip(self.rhos, self.sol_objs)])

        fig, ax = self.new_figure()
        ax.loglog(datafit_values[:, 1], dfield_values[:, 1], '--o', linewidth=0.25)
        if target_rho:
            idx = np.argmin(np.abs(np.array(self.rhos)-target_rho))
            ax.plot(datafit_values[idx, 1], dfield_values[idx, 1], 'ro')
        if optimal:
            closest_idx, closest_rho = self.optimise_lcurve(datafit_values, dfield_values)
            ax.plot(datafit_values[closest_idx, 1], dfield_values[closest_idx, 1], 'mo')
        else:
            closest_rho = None
        ax.set_xlabel("Data Fit")
        ax.set_ylabel("Diff Field")
        plt.show()

        return closest_rho

    def optimise_lcurve(self, datafit, gradientfit, origin=None):
        if origin is None:
            origin = np.array([0,0])
        else:
            origin = np.array(origin)
        xy = np.array(list(zip(datafit[:, 1], gradientfit[:, 1])))
        distances = [np.linalg.norm(xyi - origin) for xyi in xy]
        gmdistances = [2*np.sqrt(np.prod(xyi)) for xyi in xy]
        totals = 1/np.sum([gmdistances, distances], axis=0)
        closest_idx = np.argmax(totals)
        closest_rho = self.rhos[closest_idx]
        print(f"{closest_rho}\n is the closest")
        return closest_idx, closest_rho

    @staticmethod
    def __ignore(array, indices):
        return np.array([elem for i, elem in enumerate(array) if i not in indices])

    @lru_cache(maxsize=128)
    def make_confidence(self, parameter, problem=0):
        H, g = hessian(self.fitter.inner_objectives[problem].inner_criterion, parameter)
        return Function('hfn', self.fitter.inner_objectives[problem].input_list, [H])

    def calculate_confidence(self, target_rho, problem=0):
        ps_end = self.p_of(target_rho, problem)
        fisher = []
        for p in self.fitter.models[problem].ps:
            hfn = self.make_confidence(p, problem)
            fisher.append(float(hfn(
                self.fitter.models[problem].observation_times,
                *argsplit(self.fitter.problems[problem].cache.results[tokey(target_rho, ps_end)].x,
                          self.nState),
                *ps_end,
                *self.fitter.inner_objectives[problem].generate_collocation_matrix(
                    self.context.datasets[problem], self.fitter.models[problem]
                ),
                *self.fitter.inner_objectives[problem].pad_observations(
                    self.context.datasets[problem]['y']
                ),
                target_rho
            )))
        return fisher

    def draw_confidence(self, target_rho, problem=0, labels=None, ignore=None, verbose=False):
        """Plots the confidence interval estiamtes based on 'Fisher information' (curvature)"""
        ps_end = self.p_of(target_rho, problem)
        fisher = self.calculate_confidence(target_rho, problem)
        if verbose:
            print(fisher)
        pidx = range(len(ps_end))
        fig, ax = self.new_figure()
        ax.bar(pidx, ps_end)
        if ignore is None:
            ax.errorbar(pidx, ps_end, yerr=3*np.sqrt(1/np.array(fisher)), capsize=7,
                        markeredgewidth=2, linestyle='None', ecolor='k', color='k')
        else:
            ax.errorbar(self.__ignore(pidx, ignore), self.__ignore(ps_end, ignore),
                        yerr=3*np.sqrt(1/self.__ignore(np.array(fisher), ignore)),
                        capsize=7, markeredgewidth=2, linestyle='None', ecolor='k', color='k')
        if labels:
            ax.set_xticks(pidx)
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks(pidx)
            ax.set_xticklabels([str(p).replace('_', '') for p in self.fitter.models[problem].ps])
        plt.title("Confidence Interval Estimates using Fisher Information")
        plt.show()

    def validate_on_confidence(self, problem=0):
        distances = []
        was_inf = []
        for rho in self.rhos:
            fisher = 3*np.sqrt(1/np.array(self.calculate_confidence(rho, problem)))
            distances.append(max(fisher[[not (nn or nf) for nn, nf in zip(np.isnan(fisher), np.isinf(fisher))]]))
            was_inf.append(any(np.isinf(fisher)))
        was_real = [not w for w in was_inf]
        plt.loglog(np.array(self.rhos)[was_real], np.array(distances)[was_real], 'bo')
        plt.loglog(np.array(self.rhos)[was_inf], np.array(distances)[was_inf], 'ro')
        plt.title(r"Maximum $3\sigma$ Confidence Interval Size (ignoring inf)")
        plt.xlabel(r"$\rho$")
        plt.legend(['bounded intervals', 'unidentifiable'], loc="best", bbox_to_anchor=(1.01, 1))
        plt.show()

    def draw_error(self, indexkeys, target_rho, problem=0):
        """Plots a representation of how much data fit error there is

        indexkeys: {index: datakey}
        """
        getx = Function("getx", [self.fitter.models[problem].ts,
                                 *self.fitter.models[problem].cs],
                        self.fitter.models[problem].xs)
        problem_obj = self.fitter.problems[problem]
        c_end = problem_obj.cache.results[tokey(target_rho, self.p_of(target_rho, problem))].x
        times = self.fitter.models[problem].observation_times
        xs_end = np.array([np.array(i) for i in getx(times, *argsplit(c_end, self.nState))])
        collocators = self.fitter.inner_objectives[problem].generate_collocation_matrix(
            self.context.datasets[problem], self.fitter.models[problem]
        )
        observables = [collocators[i]@xs_end[i] for i in indexkeys.keys()]
        errs = [np.abs(observable.T - np.array(self.context.datasets[problem][datakey])).reshape(-1,)
                for observable, datakey in zip(observables, indexkeys.values())]
        if len(indexkeys) == 2:
            i0s = xs_end[list(indexkeys.keys()), :]
            scales = [np.mean(err) + 1.96*np.std(err) for err in errs]
            circles = [i0 + iscl*fn(np.linspace(0, 2*np.pi, 50))
                       for i0, iscl, fn in zip(i0s, scales, [np.cos, np.sin])]
            fig, (daxis, axis) = self.new_figure([2, 1])
            daxis.plot(*i0s)
            daxis.plot(*[self.context.datasets[problem][datakey]
                         for datakey in indexkeys.values()], 'o--', alpha=0.55)
            daxis.plot(circles[0], circles[1], 'ko', alpha=0.2)
        else:
            fig, axis = self.new_figure()
        for err in errs:
            axis.plot(self.context.datasets[problem]['t'], err)
        plt.show()

    def draw_basis(self, target_rho, problem=0, labels=None):
        bfn = Function('basis_fns', [self.fitter.models[problem].ts],
                       [self.fitter.models[problem].basis])
        getx = Function("getx", [self.fitter.models[problem].ts,
                                 *self.fitter.models[problem].cs],
                        self.fitter.models[problem].xs)
        problem_obj = self.fitter.problems[problem]
        c_end = problem_obj.cache.results[tokey(target_rho, self.p_of(target_rho, problem))].x
        times = self.fitter.models[problem].observation_times
        xs_end = np.array([np.array(i) for i in getx(times, *argsplit(c_end, self.nState))])
        fig, ax = self.new_figure()
        ax.plot(self.fitter.models[problem].observation_times,
                np.abs(np.hstack([x/max(abs(x)) for x in xs_end])),
                linewidth=5)
        ax.plot(self.fitter.models[problem].observation_times,
                bfn(self.fitter.models[problem].observation_times), '--')
        for t in self.context.datasets[problem]['t']:
            ax.axvline(x=t, color='m', linewidth=0.25, linestyle='--')
        if labels:
            ax.legend(labels, loc="best", bbox_to_anchor=(1.01, 1))
        plt.show()
