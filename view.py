#!/usr/bin/env python
import glob
import warnings
from pathlib import Path
import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("error")

def read_directory(directory, data):
    appended = data
    for outfile in glob.glob(directory + "/*.out"):
        file_result = pd.read_csv(outfile, header=None, squeeze=True)
        named_result = pd.DataFrame({Path(outfile).stem: file_result})
        appended = pd.concat([appended, named_result], axis=1)
    return appended


def show_results(data, axis):
    N = len(data)
    xs = np.arange(N)
    cols = len(data.columns)
    width = 1/(cols+1)

    for idx in range(cols):
        ps = np.abs(data.iloc[:, idx])
        axis.bar(xs+width*idx, ps, width)

    axis.set_xticks(xs+width*cols/2)
    axis.set_xticklabels(xs)


@click.command()
@click.option('-d', '--results-directory', help='path to directory that holds the results')
def main(results_directory):
    data = pd.DataFrame()
    data = read_directory(results_directory, data)
    fig, ax = plt.subplots()
    show_results(data, ax)
    plt.show()

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
