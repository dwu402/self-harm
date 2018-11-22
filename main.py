#!/usr/bin/env python
import click
import ingestor
import model
import fitter
import display


def standard_integrate(model_function, context):
    """Integrates a model with the provided parameters"""
    results = model.integrate_model(model_function,
                                    context['initial_values'],
                                    context['time_span'],
                                    context['parameters'])
    return results

def model_fit(model, context):
    """Perform parameter fitting on a model"""
    fitting_results = fitter.fitter(model, context)
    return fitting_results


@click.command()
@click.option('-f', '--fit-model', is_flag=True, help='whether or not to execute fitting')
@click.option('-m', '--model-file', help='path to the model file')
@click.option('-p', '--parameter-file', help='path to the parameter file')
@click.option('-c', '--config-file', help='path to the config file')
def main(fit_model, model_file, parameter_file, config_file):
    """Control Function for Workflow"""

    model_function = ingestor.get_model(model_file)

    model_context = {}
    ingestor.get_config(model_context, config_file)
    ingestor.get_parameters(model_context, parameter_file)
    if not fit_model:
        model_results = standard_integrate(model_function, model_context)
        display.plot_trajectory(model_results['y'])
        return
    ingestor.get_data(model_context)
    fitting_results = model_fit(model_function, model_context)
    display.display_parameters(fitting_results)


if __name__ == "__main__":
    # disable pylint parsing returning error due to click argument mismatch
    main() # pylint: disable=no-value-for-parameter
