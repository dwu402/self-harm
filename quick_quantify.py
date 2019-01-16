#!/usr/bin/env python
from glob import glob
from os import path
import click
import numpy as np


def generate_glob(file_path):
    if path.isdir(file_path):
        return glob(path.join(file_path,'*.out'))
    if path.isfile(file_path):
        return glob(file_path)
    raise TypeError('Unknown file/directory type')

def read_results(results_file):
    with open(results_file) as outfile:
        runs = filter(None, outfile.read().split('---'))
        results = [filter(None, p.rstrip().split('\nf ')) for p in runs]
        parameters = [[float(p) for p in ps] for ps in results]
    return parameters


def compute_statistics(results):
    p_separated_list = [list(i) for i in zip(*results)]
    means = [np.mean(i) for i in p_separated_list]
    stds = [np.std(i) for i in p_separated_list]
    return zip(means, stds)


def display_statistics(statistics, output_file):
    if output_file is not None:
        output_stream = open(output_file, 'a')
    else:
        output_stream = None
    print('---', file=output_stream)
    for parameter in statistics:
        output_string = f'{parameter[0]} ~({parameter[1]})'
        print(output_string, file=output_stream)

@click.command()
@click.option('-f', '--results-file', help='file or directory containing results')
@click.option('-o', '--output-file', default=None, help='output file')
def main(results_file, output_file):
    glob_path = generate_glob(results_file)
    for result_file in glob_path:
        results = read_results(result_file)
        statistics = compute_statistics(results)
        display_statistics(statistics, output_file)


if __name__ == '__main__':
    main()  #pylint: disable=no-value-for-parameter
