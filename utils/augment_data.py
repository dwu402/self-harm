import numpy as np
from glob import glob
import pandas as pd
import click

@click.command()
@click.option('-f', '--files', multiple=True, help='file names')
@click.option('-c', '--categories', multiple=True, help='categories to combine')
@click.option('-o', '--output', default='data.o.csv', help='output file name')
def main(files, categories, output):
    tkey = 'Day Post Infection'

    datas = []
    ts = []
    output_df = pd.DataFrame()
    for file in files:
        for fl in glob(file):
            datas.append(pd.read_csv(fl))
            ts.append(datas[-1][tkey].iloc[-1])
    for category in categories:
        category_data = []
        for t in range(max(ts)):
            n = 0
            dv = 0
            for data in datas:
                if t in data[tkey]:
                    n += 1
                    dv += data[category][data[tkey].index[t]]
            if n == 0:
                category_data.append(np.nan)
            else:
                category_data.append(dv/n)
        output_df[category] = category_data

    output_df.to_csv(output, index=False)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
