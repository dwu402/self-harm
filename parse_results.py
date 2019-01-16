#!/usr/bin/env python
import re
from itertools import zip_longest
import numpy as np
from matplotlib import pyplot as plt
import click


def split_by(pattern, string, clean=False):
    split_string = re.compile(pattern).split(string)
    if clean:
        return filter(None, split_string)
    return split_string


@click.command()
@click.option('-f', '--results-file', help='file containing the output from the fitting')
def main(results_file):
    with open(results_file) as file_buffer:
        contents = file_buffer.read()
    splitting_pattern = r"Time span of fitting:\s.*\nInitial values:\s.*\n"
    results_per_dataset = split_by(splitting_pattern, contents, clean=True)
    single_result_pattern = r"Optimization terminated [un]{0,2}successfully.\n\s+Current function value:\s+(.*)\n\s+Iterations:\s+(.*)\n\s+Function evaluations:\s+(.*)\n"
    for result in results_per_dataset:
        singleton_results = split_by(single_result_pattern, result)[:-1]
        del singleton_results[0::4]
        temp = [iter(singleton_results)] * 3
        grouped_results = zip_longest(*temp, fillvalue="")
        grouped_results = np.array(list(grouped_results)).astype(np.float)
        fig, ax = plt.subplots()
        ax.boxplot([r[0] for r in grouped_results], positions=[1])
        ax_twin = ax.twinx()
        ax_twin.boxplot(np.array([r[1:] for r in grouped_results]), positions=[2,3])
        ax.set_xlim(0, 4)
        plt.show()


if __name__ == "__main__":
    main()  #pylint: disable=no-value-for-parameter
