#!/usr/bin/env python
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
    fitting_results = fitter.fitter(context)
    return fitting_results


def show_trajectory_and_fitting_results(context, fitting_results):
    parameter_strings = fitting_results.get_parameters().strip().split('\n')
    parameters = [float(p) for p in parameter_strings]
    fitted_traj = model.integrate_model(context['model'],
                                        context['initial_values'],
                                        context['time_span'],
                                        parameters)
    canvas = display.new_canvas()
    display.plot_trajectory(fitted_traj, canvas, show=False)
    display.show_data(context, canvas, show=True)


@click.command()
@click.option('-a', '--action', default='integrate', help='action to take: [i]ntegrate, [s]how-data, [f]it')
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
    elif action in ['f', 'fit']:
        ingestor.get_data(model_context)
        fitting_results = model_fit(model_context)
        if verbose or not output_file:
            display.display_parameters(fitting_results)
        if output_file:
            display.write_results(fitting_results, output_file)
        if verbose > 1:
            show_trajectory_and_fitting_results(model_context, fitting_results)

    else:
        print('Action not found: ', action)

if __name__ == "__main__":
    # disable pylint parsing returning error due to click argument mismatch
    main() # pylint: disable=no-value-for-parameter
