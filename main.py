import click
import model
import fitter


@click.command()
def main():
    model_function = model.get_model('model_file')
    model_context = {
        'initial_values': [0, 1],
        'time_span': model.np.linspace(0, 10, 10000),
        'parameters': [2, 3]
    }
    model_results = model.integrate_model(model_function, model_context)
    fitter.plot_results(model_results)

if __name__ == "__main__":
    main()
