#!/usr/bin/env python
"""Helper application that views results"""
import warnings
import click
import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("error")


def parse_outfile(file_name):
    with open(file_name) as outfile:
        runs = filter(None, outfile.read().split('---'))
        results = [filter(None, p.rstrip().split('\nf ')) for p in runs]
        parameters = [[float(p) for p in ps] for ps in results]
    return parameters


def bar_results(data, axis):
    N = len(data)
    xs = np.arange(N)
    cols = len(data[0])
    width = 1/(cols+1)

    for idx in range(cols):
        ps = data[idx]
        axis.bar(xs+width*idx, ps, width)

    axis.set_xticks(xs+width*cols/2)
    axis.set_xticklabels(xs)


def box_results(data, axis):
    transposed_data = [list(i) for i in zip(*data)]
    axis.boxplot(transposed_data, showfliers=True)


def show_results(plot_type, data, axis):
    # data = np.log(data)
    if plot_type == 'bar':
        bar_results(data, axis)
    elif plot_type == 'box':
        box_results(data, axis)
    else:
        raise TypeError('Plot type' + str(type) + 'not recognised')


@click.command()
@click.option('-f', '--results-file', help='path to file that holds the results')
@click.option('-t', '--plot-type', default='box', help='type of plot')
def main(results_file, plot_type):
    data = parse_outfile(results_file)
    fig, ax = plt.subplots()
    show_results(plot_type, data, ax)
    plt.show()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
