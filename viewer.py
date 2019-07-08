import warnings
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
            ax.quiver(xs_end[xaxis], xs_end[yaxis],
                      spline_dfield[:, xaxis], spline_dfield[:, yaxis],
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
                target_rho)
                + 2*self.context.fitting_configuration['regularisation_parameter'][2]
            ))
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

    def validate_on_confidence(self, problem=0, method='absolute'):
        r""" Implemented methods:
            absolute: ||e||_\infty
            relative: ||e/x||_\infty
            uniform: ||e/(1+x)||_\infty
        """
        distances = []
        was_inf = []
        was_nan = []
        for rho in self.rhos:
            intervals = 3*np.sqrt(1/np.array(self.calculate_confidence(rho, problem)))
            if method == "absolute":
                error = intervals
            elif method == "relative":
                error = intervals/self.p_of(rho, problem)
            elif method == "uniform":
                error = intervals/(1+self.p_of(rho, problem))
            distances.append(max(error[[not (nn or nf)
                                        for nn, nf in zip(np.isnan(error), np.isinf(error))]]))
            was_inf.append(any(np.isinf(error)))
            was_nan.append(any(np.isnan(error)))
        was_real = [(not wi and not wn) for wi, wn in zip(was_inf, was_nan)]
        fig, ax = self.new_figure()
        ax.loglog(np.array(self.rhos)[was_real], np.array(distances)[was_real], 'bo')
        ax.loglog(np.array(self.rhos)[was_inf], np.array(distances)[was_inf], 'rX')
        ax.loglog(np.array(self.rhos)[was_nan], np.array(distances)[was_nan], 'md')
        ax.set_title(r"Maximum $3\sigma$ Confidence Interval Size (ignoring infs)")
        ax.set_xlabel(r"$\rho$")
        ax.legend(['bounded intervals', 'unidentifiable', 'does not minimize'],
                   loc="best", bbox_to_anchor=(1.01, 1))
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
        collocator_index_map = {v: i for i, v in
                                enumerate(self.context.fitting_configuration['observation_vector'])}
        observables = [collocators[collocator_index_map[i]]@xs_end[i] for i in indexkeys.keys()]
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
            axis.legend(indexkeys.values())
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

    @staticmethod
    def __replace(iterable, location, value):
        return np.array(list(iterable[:location]) + [value] + list(iterable[location+1:]))

    def generate_profile(self, target_rho, problem=0, parameter=0, irange=None, lower=None, upper=None, axes=None):
        """ Generate the parameter profile around the optimal solution for a given parameter """
        initial_estimate = self.fitter.solutions[str(target_rho)][problem].x
        interest_parameter = initial_estimate[parameter]
        # Construct the range for profiling
        if lower is None and upper is None:
            if irange is None:
                raise TypeError("generate_profile expects either irange or lower and upper")
            parameter_range = np.linspace(interest_parameter-irange/2, interest_parameter+irange/2)
        elif lower is not None and upper is not None:
            if lower > interest_parameter or upper < interest_parameter:
                warnings.warn("Interest parameter is outside the lower/upper range")
            parameter_range = np.linspace(lower, upper)
        # Reset value of the nuisance spline parameters
        original_spline_ps = self.fitter.problems[problem].cache.get(tokey(target_rho, initial_estimate)).x
        # Solve over profile
        profile = []
        for p in parameter_range:
            self.fitter.problems[problem].cache.recent = original_spline_ps
            modded_ps = self.__replace(initial_estimate, parameter, p)
            profile.append(self.fitter.problems[problem].function(modded_ps))
        # Plotting
        if axes is None:
            fig, axes = self.new_figure()
        axes.plot(parameter_range, profile)
        axes.axvline(interest_parameter)
        axes.plot(interest_parameter, self.fitter.problems[problem].cache.get(tokey(target_rho, initial_estimate)).fun, 'ro')

    @staticmethod
    def __make_plot_grid(number):
        """ Intelligently determine num rows and num columns for a given number of subplots requirement """
        n1 = np.ceil(np.sqrt(number))
        n2 = np.ceil(number/n1)
        return int(n1), int(n2)

    @staticmethod
    def __isiterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    def generate_all_profiles(self, target_rho, problem=0, irange=None, lower=None, upper=None):
        """ Generate profiles over all parameters """
        num_params = self.context.modelling_configuration['model_form']['parameters']
        # setup plotting canvas
        num_rows, num_cols = self.__make_plot_grid(num_params)
        fig, axes = self.new_figure(with_axes={"nrows":num_rows, "ncols":num_cols})
        for i, ax in zip(range(num_params), axes.flatten()):
            if self.__isiterable(irange):
                self.generate_profile(target_rho, problem=problem, parameter=i, irange=irange[i], axes=ax)
            elif self.__isiterable(lower) and self.__isiterable(upper):
                self.generate_profile(target_rho, problem=problem, parameter=i, lower=lower[i], upper=upper[i], axes=ax)
            else:
                self.generate_profile(target_rho, problem=problem, parameter=i, irange=irange, lower=lower, upper=upper, axes=ax)
        fig.suptitle("Parameter Profiles (Objective Function)")
        plt.show()

    def compare_parameters(self, target_rho, categories=None, labels=None):
        """Plots all parameters on a bar graph next to each other
            categories: integer bins
        """
        from cycler import cycler

        all_params = np.stack([s.x for s in self.fitter.solutions[str(target_rho)]])
        num_problems, num_params = all_params.shape
        fig, ax = self.new_figure()
        if categories:
            colour_list = [x['color'] for x in list(plt.rcParams['axes.prop_cycle'])]
            colours = cycler(color=[colour_list[i%len(colour_list)] for i in categories])
            ax.set_prop_cycle(colours)
        rel_locs = np.linspace(-0.3, 0.3, num_problems)
        width = 0.6/num_problems
        for delta, problem_params in zip(rel_locs, all_params):
            locs = np.array(range(num_params)) + delta
            ax.bar(locs, problem_params, width)
        if labels:
            ax.set_xticks(range(num_params))
            ax.set_xticklabels(labels[:num_params])
        plt.show()
