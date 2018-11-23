import numpy as np


def select_data(raw_data):
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


def scale_data(data):
    """Scale data onto the correct scale for the variable"""
    for col in ['x', 'z']:
        first_val = data[col][0]
        max_dist = max(np.abs(data[col] - first_val))
        scaled_vals = [(val - first_val)/max_dist for val in data[col]]
        data[col] = scaled_vals

def threshold_data(data):
    """Throw out data that is below a certain RBC threshold"""
    threshold_value = 0.25
    values = data['x']
    thresholded_index = next((values.index(x) for x in values if x > threshold_value), 0)
    for col in ['x', 'z']:
        data[col] = data[col][thresholded_index:]

def treat_data(raw_data):
    clean_data = select_data(raw_data)
    scale_data(clean_data)
    threshold_data(clean_data)
    return clean_data

def error_fn(data, fit):
    return np.Infinity
