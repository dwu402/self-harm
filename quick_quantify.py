#!/usr/bin/env python
"""Specific to the generic_immune model"""

from glob import glob
from os import path
import re
from itertools import zip_longest
import click
import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler


def generate_glob(file_path):
    """Returns a glob of all readable files, and whether or not it is a directory"""
    if path.isdir(file_path):
        return glob(path.join(file_path, '*.out')), True
    if path.isfile(file_path):
        return glob(file_path), False
    raise TypeError('Unknown file/directory type')

def read_results(results_file):
    """Reads in a file that is pseudo-YAML delimited"""
    with open(results_file) as outfile:
        contents = outfile.read()
    runs = filter(None, contents.split('---'))
    results = [filter(None, p.rstrip().split('\nf ')) for p in runs]
    parameters = [[float(p) for p in ps] for ps in results]
    return parameters


def split_by(pattern, string, clean=False):
    """Split a string by a regex. Clean split returns a filter object without empty strings"""
    split_string = re.compile(pattern).split(string)
    if clean:
        return filter(None, split_string)
    return split_string


def read_summary(summary_file):
    """Reads the stdout file that contains objective function value information"""
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
    """Removes a proportion of results based on objective function values"""
    thresholds = np.percentile(summary_notes, 25, axis=0)
    threshold = thresholds[0]
    acceptable_indices = [i for i, val in enumerate(summary_notes) if val[0] < threshold]
    return [val for i, val in enumerate(results) if i in acceptable_indices]


def standard_deviation(sample, mean):
    """Shitty implementation of the standard deviation"""
    variance = sum([(s-mean)**2 for s in sample])/len(sample)
    return np.sqrt(variance)


def compute_statistics(results):
    """Returns the mean and std deviation of a list"""
    p_separated_list = [list(i) for i in zip(*results)]
    means = [np.mean(val) for val in p_separated_list]
    stds = [np.std(val)/len(val) for val in p_separated_list]
    return zip(means, stds)


def trim_data(results, summary_notes):
    """Wrapper for removing unanted results"""
    if summary_notes != []:
        return remove_bad_results(summary_notes, results)
    return results


def plot_statistics(statistics):
    """Plotting function, unused"""
    fig, ax = plt.subplots()
    stats = list(statistics)
    x = np.arange(len(stats))
    h = [p[0] for p in stats]
    e = [p[1] for p in stats]
    ax.bar(x, h, yerr=e)


def plot_data(data):
    """Other plotting function"""
    fig, ax = plt.subplots()
    p_list = np.array(data)
    ax.violinplot(p_list, showmedians=True, showextrema=False)
    # specific to the immune system
    ax.set_xticks(np.arange(1, 10))
    ax.set_xticklabels(['r','k','p','s','d','f','g','j','l'])
    # ax.set_yscale("log", nonposy='clip')
    # ax.set_ylim(0, 10)


def display_statistics(statistics, output_file):
    """Writes the calcualted stats to stdout or file, as specified"""
    if output_file is not None:
        output_stream = open(output_file, 'a')
    else:
        output_stream = None
    print('---', file=output_stream)
    for parameter in statistics:
        output_string = f'{parameter[0]} ~({parameter[1]})'
        print(output_string, file=output_stream)


def read_output_file(output_file):
    """Reads the output file of this program for furhter analysis"""
    with open(output_file) as of_handle:
        outputs = of_handle.read()
    chunks = filter(None, outputs.split('---'))
    comparison_stats = []
    stats_pattern = r"(.*) \~\((.*)\)"
    for chunk in chunks:
        stats = filter(None, chunk.split('\n'))
        stats_array = [tuple(split_by(stats_pattern, statistic, clean=True)) for statistic in stats]
        comparison_stats.append(stats_array)
    return comparison_stats


def display_comparison(output_file, xlabels):
    """Plots the comparison between different groups of stats calculated"""
    comparisons = read_output_file(output_file)
    comparisons = np.array(comparisons).astype('float')
    xs = np.arange(0,len(xlabels))
    plt.rc('axes', prop_cycle=cycler('color', ['g','g','g','r','r','r']))
    fig, ax = plt.subplots()
    for stat in comparisons:
        means = np.array([j[0] for j in stat])
        stdvs = np.array([j[1] for j in stat])
        ax.errorbar(xs, means, yerr=stdvs, fmt="o")
    ax.set_title('Mean values for each parameter')
    ax.set_xticks(np.arange(0, len(xlabels)))
    ax.set_xticklabels(xlabels)
    # ax.set_yscale("log", nonposy='clip')


def do_quirky(output_file):
    """Looking at eigenvalues (plotter)"""
    results = read_output_file(output_file)
    results = np.array(results).astype('float')
    plt.rc('axes', prop_cycle=cycler('color', ['g','g','g','r','r','r']))
    fig, ax = plt.subplots()
    for result in results:
        eigens = compute_eigenvalues([r[0] for r in result])
        ax.plot(eigens, 'o')


def compute_eigenvalues(p_list):
    """Implementation to calculate the eigenvalues of the generic immune model from a lsit of the parameters"""
    characteristic_polynomial = np.array([1, -p_list[3]/p_list[4], 0, 1])
    gammas = np.roots(characteristic_polynomial)
    gamma_max = np.real(max(gammas[np.isreal(gammas)]))

    return [p_list[0] - p_list[1]*gamma_max,
            3*p_list[4]/(1+gamma_max**3) - p_list[4],
            p_list[5]*gamma_max - p_list[6]]


def compute_eigen_stats(results):
    """Go throught the individual results and calcuate the eigenvalues for further analysis"""
    eigenvalues = [compute_eigenvalues(result) for result in results]
    sorted_list = [list(i) for i in zip(*eigenvalues)]
    means = [np.mean(val) for val in sorted_list]
    stds = [np.std(val)/len(val) for val in sorted_list]
    return zip(means, stds)


@click.command()
@click.option('-f', '--results-file', help='file or directory containing results')
@click.option('-s', '--summary-file', default=None, help='file that contains the summary notes')
@click.option('-g', '--graphical', is_flag=True, help='plot the results instead of printing')
@click.option('-c', '--compare', is_flag=True, help='compare the results between different runs')
@click.option('-q', '--quirky', is_flag=True, help='weird flag, meant for testing')
@click.option('-o', '--output-file', default=None, help='output file')
def main(results_file, summary_file, graphical, compare, quirky, output_file):
    glob_path, isdir = generate_glob(results_file)
    if not isdir or not summary_file:
        summary_notes = [[]]*len(glob_path)
    else:
        summary_notes = read_summary(summary_file)
    if compare and not output_file:
        output_file = "comparison_summary.results"
    if quirky:
        xlabels = ['r-kY', '3d/(1+Y^2)', 'fY-g']
        compare = True
    else:
        xlabels = list('rkpsdfgjl')
    if output_file:
        f = open(output_file, 'w')
        f.close()
    for index, result_file in enumerate(glob_path):
        results = read_results(result_file)
        trimmed_data = trim_data(results, summary_notes[index])
        if graphical:
            plot_data(trimmed_data)
        else:
            if quirky:
                statistics = compute_eigen_stats(trimmed_data)
            else:
                statistics = compute_statistics(trimmed_data)
            display_statistics(statistics, output_file)
    if compare:
        display_comparison(output_file, xlabels)
    plt.show()


if __name__ == '__main__':
    main()  #pylint: disable=no-value-for-parameter
