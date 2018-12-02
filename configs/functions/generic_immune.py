import numpy as np
from scipy.interpolate import interp1d

def select_data(raw_data):
    data_cols = {
        't': 'Day Post Infection',
        'x': 'PD',
        'z': 'RBC'
    }
    clean_data = dict.fromkeys(data_cols.keys())
    for key in data_cols:
        clean_data[key] = raw_data[data_cols[key]]

    return clean_data


def scale_data(data):
    """Scale data onto the correct scale for the variable"""
    for col in ['x', 'z']:
        first_val = data[col][0]
        max_dist = max(np.abs(data[col] - first_val))
        scaled_vals = [(val - first_val)/max_dist for val in data[col]]
        data[col] = np.array(scaled_vals)


def threshold_data(data):
    """Throw out data that is below a certain RBC threshold"""
    threshold_value = 0.01
    values = list(data['x'])
    accepted_values = [v > threshold_value for v in values]
    for col in data.keys():
        data[col] = np.array([i for i,j in zip(data[col], accepted_values) if j])


def treat_data(context, raw_data):
    clean_data = select_data(raw_data)
    scale_data(clean_data)
    threshold_data(clean_data)

    # Modify the time span of integration to match the data
    context['time_span'] = [clean_data['t'][0], clean_data['t'][-1], len(clean_data['t'])]
    context['initial_values'] = [clean_data['x'][0], 0, clean_data['z'][0]]
    context['data'] = clean_data

    print("Time span of fitting: ", context['time_span'])
    print("Initial values: ", context['initial_values'])


def error_fn(data, fit, parameters):
    beta = 0.25

    fit_x = interp1d(fit['t'], np.array([x[0] for x in fit['y']]))
    fit_z = interp1d(fit['t'], np.array([z[2] for z in fit['y']]))
    distance = np.linalg.norm([(data['x'] - fit_x(data['t'])) + (data['z'] - fit_z(data['t']))])
    regularisation = np.linalg.norm(parameters) + np.linalg.norm([p for p in parameters if p < 0])
    return distance + beta*regularisation


def data_plot(data):
    x = data['t']
    ys = [(data['x'][i], data['z'][i]) for i in range(len(x))]

    return {'x': data['x'], 'y': data['z']}
