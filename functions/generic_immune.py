import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def parse_royal(context, raw_datasets):
    return parse(context, raw_datasets, select_data_royal)

def parse_torres(context, raw_datasets):
    return parse(context, raw_datasets, select_data_torres)

def parse(context, raw_datasets, selection_function):
    threshold_value = 1e-5
    clean_datasets = []
    updates = {'initial_values': [],
               'time_span': [],
               'observation_vector': None,
              }

    for data in raw_datasets:
        selected_data = selection_function(data)
        # threshold out low pathogen levels
        vals = selected_data['x']
        acceptable_vals = [v > threshold_value for v in vals]
        thresholded_data = selected_data[acceptable_vals]
        clean_datasets.append(thresholded_data.reset_index())

    ics_z = np.array([dataset['z'] for dataset in clean_datasets]).flatten()
    for dataset in clean_datasets:
        # shift initial conditions
        dataset['z'] = dataset['z'] - max(ics_z)

        # create the context updates
        updates['initial_values'].append([dataset['x'].iloc[0], 0, dataset['z'].iloc[0]])
        dataset['t'] = dataset['t'] - dataset['t'].iloc[0]
        updates['time_span'].append([0, dataset['t'].iloc[-1], dataset['t'].iloc[-1]])

        dataset['y']  = pd.Series([v for v in dataset[['x', 'z']].values], index = dataset.index)
    updates['observation_vector'] = np.array([0, 2])

    return clean_datasets, updates

def select_data_torres(data):
    """Map the data columns to model state variables"""
    data_cols = {
        'Day Post Infection': 't',
        'PD': 'x',
        'RBC': 'z',
        'Status': 'status'
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def select_data_royal(data):
    """Map the data columns to model state variables"""
    data_cols = {
        'day': 't',
        'parasite': 'x',
        'rbc': 'z',
        'weight': 'w',
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def visualise():
    return
