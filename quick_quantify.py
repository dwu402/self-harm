#!/usr/bin/env python
from glob import glob
from os import path
import re
from itertools import zip_longest
import click
import numpy as np
from matplotlib import pyplot as plt


def generate_glob(file_path):
    if path.isdir(file_path):
        return glob(path.join(file_path,'*.out')), True
    if path.isfile(file_path):
        return glob(file_path), False
    raise TypeError('Unknown file/directory type')

def read_results(results_file):
    with open(results_file) as outfile:
        contents = outfile.read()
    runs = filter(None, contents.split('---'))
    results = [filter(None, p.rstrip().split('\nf ')) for p in runs]
    parameters = [[float(p) for p in ps] for ps in results]
    return parameters


def split_by(pattern, string, clean=False):
    split_string = re.compile(pattern).split(string)
    if clean:
        return filter(None, split_string)
    return split_string


def read_summary(summary_file):
    if not summary_file:
        return None
    summary_notes = []
    with open(summary_file) as sfh:
        summary_contents = sfh.read()
    splitting_pattern = r"Time span of fitting:\s.*\nInitial values:\s.*\n"
    results_per_dataset = split_by(splitting_pattern, summary_contents, clean=True)
    single_result_pattern = r"Optimization terminated [un]{0,2}successfully.\n\s+Current function value:\s+(.*)\n\s+Iterations:\s+(.*)\n\s+Function evaluations:\s+(.*)\n"
    for result in results_per_dataset:
        singleton_results = split_by(single_result_pattern, result)[:-1]
        del singleton_results[0::4]
        temp = [iter(singleton_results)] * 3
        grouped_results = zip_longest(*temp, fillvalue="")
        grouped_results = np.array(list(grouped_results)).astype(np.float)
        summary_notes.append(grouped_results)
    return summary_notes


def remove_bad_results(summary_notes, results):
    thresholds = np.quantile(summary_notes, 0.75, axis=0)
    threshold = thresholds[0]
    acceptable_indices = [i for i, val in enumerate(summary_notes) if val[0] < threshold]
    return [val for i, val in enumerate(results) if i in acceptable_indices]


def standard_deviation(sample, mean):
    variance = 0
    for x in sample:
        variance += (x-mean)**2
    return np.sqrt(variance/len(sample))


def compute_statistics(results, summary_notes):
    if summary_notes != []:
        results = remove_bad_results(summary_notes, results)
    p_separated_list = [list(i) for i in zip(*results)]
    means = [np.mean(val) for val in p_separated_list]
    stds = [np.std(val)/len(val) for val in p_separated_list]
    return zip(means, stds)


def plot_statistics(statistics):
    fig, ax = plt.subplots()
    stats = list(statistics)
    x = np.arange(len(stats))
    h = [p[0] for p in stats]
    e = [p[1] for p in stats]
    ax.bar(x, h, yerr=e)


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
@click.option('-s', '--summary-file', default=None, help='file that contains the summary notes')
@click.option('-g', '--graphical', is_flag=True, help='plot the results instead of printing')
@click.option('-o', '--output-file', default=None, help='output file')
def main(results_file, summary_file, graphical, output_file):
    glob_path, isdir = generate_glob(results_file)
    if not isdir or not summary_file:
        summary_notes = [[]]*len(glob_path)
    else:
        summary_notes = read_summary(summary_file)
    for index, result_file in enumerate(glob_path):
        results = read_results(result_file)
        statistics = compute_statistics(results, summary_notes[index])
        if graphical:
            plot_statistics(statistics)
        else:
            display_statistics(statistics, output_file)
    plt.show()


if __name__ == '__main__':
    main()  #pylint: disable=no-value-for-parameter
