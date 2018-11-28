"""Data wrangling for the default sources"""
import pandas as pd
from pathlib import Path
import click


@click.command()
@click.option('-f', '--file-path', help='path to file')
def main(file_path):
    file_path = Path(file_path)
    data_contents = pd.read_excel(file_path)
    key_column = data_contents['Mouse']
    for key in key_column.unique():
        subfile_path = file_path.parent.joinpath('unique_key' + str(key) + '.csv')
        data_subsection = data_contents[key_column == key]
        data_subsection.to_csv(subfile_path, mode='w', index=False)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
