import click
import model
import fitter


@click.command()
@click.option('-f', '--model-file', help='path to the model file')
@click.option('-p', '--parameter-file', help='path to the parameter file')
def main(model_file, parameter_file):
    """Control Function for Workflow"""

    model_function = model.get_model(model_file)
    model_context = model.get_context(parameter_file)
    model_results = model.integrate_model(model_function, model_context)

    fitter.plot_results(model_results['y'])

if __name__ == "__main__":
    main()
