import numpy as np
from scipy.interpolate import interp1d

def parse(context, raw_datasets):
    threshold_value = 1e-5
    clean_datasets = []
    updates = {'initial_values': [],
               'time_span': [],
               'smoothed_data': [],
              }

    for data in raw_datasets:
        selected_data = select_data(data)
        # threshold out low pathogen levels
        vals = selected_data['x']
        acceptable_vals = [v > threshold_value for v in vals]
        thresholded_data = selected_data[acceptable_vals]
        clean_datasets.append(thresholded_data.reset_index())

    ics_z = [dataset['z'].iloc[0] for dataset in clean_datasets]
    for dataset in clean_datasets:
        # shift initial conditions
        dataset['z'] = dataset['z'] - max(ics_z)

        # create the context updates
        updates['initial_values'].append([dataset['x'].iloc[0], 0, dataset['z'].iloc[0]])
        dataset['t'] = dataset['t'] - dataset['t'].iloc[0]
        updates['time_span'].append([0, dataset['t'].iloc[-1]*2, dataset['t'].iloc[-1]*2])
        #updates['smoothed_data'].append(smooth_data(context, dataset))

    return clean_datasets, updates

def select_data(data):
    """Map the data columns to model state variables"""
    data_cols = {
        'Day Post Infection': 't',
        'PD': 'x',
        'RBC': 'z',
        'Status': 'status'
    }
    return data[data_cols.keys()].rename(columns=data_cols)

def visualise():
    return
