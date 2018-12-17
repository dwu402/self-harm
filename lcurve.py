#!/usr/bin/env python
import click
from matplotlib import pyplot as plt
import numpy as np

@click.command()
@click.option('-f', '--file', help='file to read results from')
def main(file):
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
    plt.plot([r[1][0] for r in run_values], [r[1][1] for r in run_values],'-')
    plt.title('L-Curve, Beta from {} to {}'.format(min([r[0] for r in run_values]),
                                                   max([r[0] for r in run_values])))
    plt.xlabel('Log Residual')
    plt.ylabel('Log Penalty')
    # for point in run_values:
    #     ax.annotate(str(point[0]), xy=point[1], textcoords='data')
    plt.show()

if __name__=="__main__":
    main() # pylint: disable=no-value-for-parameter
