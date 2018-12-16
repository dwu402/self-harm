#!/usr/bin/env python
"""Module responsible for defining workflows in the application"""
import warnings
import click
import ingestor
import model
import fitter
import display

warnings.filterwarnings("error")

def standard_integrate(context):
    """Integrates a model with the provided parameters"""
    results = model.integrate_model(context['model'],
                                    context['initial_values'],
                                    context['time_span'],
                                    context['parameters'])
    return results

def model_fit(context):
    """Perform parameter fitting on a model"""
    loops = context['refits']
    fitter_results = fitter.FitterReturnCollection()
    for i in range(loops):
        fitter_results.add_result(fitter.fitter(context))
    return fitter_results


def compare_data_and_model(context, model_results):
    """Shows the trajectory from parameters vs data"""
    canvas = display.new_canvas()
    display.plot_trajectory(model_results, canvas, show=False)
    display.show_data(context, canvas, show=True)


def output_results(fitting_results, verbose, output_file):
    if verbose or not output_file:
        display.display_parameters(fitting_results)
    if output_file:
        display.write_results(fitting_results, output_file)
    if verbose > 1:
        show_trajectory_and_fitting_results(model_context, fitting_results)



def show_trajectory_and_fitting_results(context, fitting_results):
    """WIP. Will show the fitting results next to the data"""
    parameters = fitting_results.get_parameters()
    fitted_traj = model.integrate_model(context['model'],
                                        context['initial_values'],
                                        context['time_span'],
                                        parameters)
    canvas = display.new_canvas()
    display.plot_trajectory(fitted_traj, canvas, show=False)
    display.show_data(context, canvas, show=True)


@click.command()
@click.option('-a', '--action', default='integrate', help='action to take: [i]ntegrate, [s]how-data, [c]ompare, [f]it')
@click.option('-v', '--verbose', count=True, help='whether or not to be verbose in output (fitting only)')
@click.option('-c', '--config-file', help='path to the config file')
@click.option('-o', '--output-file', help='path to output results to')
def main(action, verbose, config_file, output_file):
    """Control Function for Workflow"""
    model_context = ingestor.initialise_context()
    ingestor.get_config(model_context, config_file)
    ingestor.get_model(model_context)
    ingestor.get_parameters(model_context)
    if action in ['i', 'integrate']:
        model_results = standard_integrate(model_context)
        display.plot_trajectory(model_results)
    elif action in ['s', 'show-data']:
        ingestor.get_data(model_context)
        display.show_data(model_context)
    elif action in ['c', 'compare']:
        ingestor.get_data(model_context)
        model_results = standard_integrate(model_context)
        compare_data_and_model(model_context, model_results)
    elif action in ['f', 'fit']:
        ingestor.get_data(model_context)
        fitting_results = model_fit(model_context)
        output_results(fitting_results, verbose, output_file)

    else:
        print('Action not found: ', action)

if __name__ == "__main__":
    # disable pylint parsing returning error due to click argument mismatch
    main() # pylint: disable=no-value-for-parameter
