import numpy as np


def treat_data(raw_data):
    mouse_number = 0
    data_cols = {
        't': 'Day Post Infection',
        'x': 'PD',
        'z': 'RBC'
    }

    mouse = raw_data['Mouse'].unique()[mouse_number]
    mouse_data = raw_data[raw_data['Mouse'] == mouse]

    clean_data = dict.fromkeys(data_cols.keys())
    for key in data_cols:
        clean_data[key] = mouse_data[data_cols[key]]

    return clean_data


def error_fn(data, fit):
    return np.Infinity
