#!/usr/bin/env python
import click
import ingestor
import model
import fitter
import display
import warnings

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


@click.command()
@click.option('-a', '--action', default='integrate', help='action to take: integrate, show-data, fit')
@click.option('-c', '--config-file', help='path to the config file')
def main(action, config_file):
    """Control Function for Workflow"""
    model_context = ingestor.initialise_context()
    ingestor.get_config(model_context, config_file)
    ingestor.get_model(model_context)
    ingestor.get_parameters(model_context)
    if action == 'integrate':
        model_results = standard_integrate(model_context)
        display.plot_trajectory(model_results)
    elif action == 'show-data':
        ingestor.get_data(model_context)
        display.show_data(model_context)
    elif action == 'fit':
        ingestor.get_data(model_context)
        fitting_results = model_fit(model_context)
        display.display_parameters(fitting_results)
    else:
        print('Action not found: ', action)

if __name__ == "__main__":
    # disable pylint parsing returning error due to click argument mismatch
    main() # pylint: disable=no-value-for-parameter
