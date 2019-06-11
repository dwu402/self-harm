import click
import ingestor
import modeller
import fitter
import numpy as np

@click.command()
@click.option('-f', '--run-file', help="file that specifies config, data and model files")
@click.option('-a', '--action', help="action to take: [i], [s], [f], [c], [l]")
@click.option('-o', '--output-file', help="file to direct output to")
@click.option('-v', '--verbose', is_flag=True, help="verbosity flag")
def main(run_file, action, output_file, verbose):
    context = ingestor.Context(run_file)
    solver = fitter.Fitter(context)
    for rho in np.logspace(*context.fitting_configuration['regularisation_parameter'][:2], num=20):
        solver.solve(rho)
        # update the initial iterate from previous rho's value
        for idx, problem in enumerate(solver.problems):
            problem.initial_guess = solver.solutions[str(rho)][idx].x
    solver.write(output_file)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
