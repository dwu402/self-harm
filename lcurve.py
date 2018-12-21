#!/usr/bin/env python
import click
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

@click.command()
@click.option('-f', '--file', help='file to read results from')
@click.option('-s', '--show', is_flag=True, help='show')
def main(file, show):
    with open(file) as fh:
        contents = fh.read()
    runs = [list(filter(None, run.strip().split('\n'))) for run in
            filter(None, contents.strip().split('---\n'))]

    run_values = []
    for run in runs:
        get_value = lambda i: float(run[i].split(':')[-1])
        run_value = (get_value(0), (np.log(get_value(1)), np.log(get_value(2))))
        run_values.append(run_value)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('plasma')
    sc = plt.scatter(x=[r[1][0] for r in run_values],
                     y=[r[1][1] for r in run_values],
                     c = [(r[0]) for r in run_values],
                     norm=colors.LogNorm(),
                     cmap=cmap)
    plt.title('L-Curve, Beta from {} to {}'.format(min([r[0] for r in run_values]),
                                                   max([r[0] for r in run_values])))
    plt.xlabel('Log Residual')
    plt.ylabel('Log Penalty')
    plt.colorbar(sc)
    if show:
        plt.show()
    return 0

if __name__=="__main__":
    main() # pylint: disable=no-value-for-parameter
