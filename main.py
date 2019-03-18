import click
import ingestor
import modeller
import fitter

@click.command()
@click.option('-f', '--run-file', help="file that specifies config, data and model files")
@click.option('-a', '--action', help="action to take: [i], [s], [f], [c], [l]")
@click.option('-o', '--output-file', help="file to direct output to")
@click.option('-v', '--verbose', is_flag=True, help="verbosity flag")
def main(run_file, action, output_file, verbose):
    context = ingestor.initialise_context()
    ingestor.read_run_file(context, run_file)
    model = modeller.Model(context)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
