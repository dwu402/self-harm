import numpy as np
from scipy.interpolate import interp1d
# import ot

import matplotlib.pyplot as plt

def select_data(raw_data):
    """Map the data columns to model state variables"""
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
    # for col in ['x', 'z']:
    #     first_val = data[col][0]
    #     max_dist = max(np.abs(data[col] - first_val))
    #     scaled_vals = [(val - first_val)/max_dist for val in data[col]]
    #     data[col] = np.array(scaled_vals)
    data['z'] = data['z'] - data['z'][0]


def threshold_data(data):
    """Throw out data that is below a certain RBC threshold"""
    threshold_value = 0.01
    values = list(data['x'])
    accepted_values = [v > threshold_value for v in values]
    for col in data.keys():
        data[col] = np.array([i for i, j in zip(data[col], accepted_values) if j])


def treat_data(context, raw_data):
    """Select, scale and treshold data; define context based on data"""
    clean_data = select_data(raw_data)
    scale_data(clean_data)
    threshold_data(clean_data)

    # Modify the time span of integration to match the data
    context['time_span'] = [clean_data['t'][0], clean_data['t'][-1]*1.05, len(clean_data['t'])*3]
    context['initial_values'] = [clean_data['x'][0], 0, clean_data['z'][0]]
    context['data'] = clean_data

    print("Time span of fitting: ", context['time_span'])
    print("Initial values: ", context['initial_values'])


def error_fn(data, fit, parameters, regularisation, detailed=False):
    """Objective function to minimize"""
    beta = regularisation

    fit_x = interp1d(fit['t'], np.array([x[0] for x in fit['y']]))
    fit_z = interp1d(fit['t'], np.array([z[2] for z in fit['y']]))

    data_points = np.column_stack((data['x'], data['z']))
    if data['t'][-1] > fit['t'][-1]:
        ts_to_fit = [t for t in data['t'] if t <= fit['t'][-1]]
    else:
        ts_to_fit = data['t']
    interpolation_penalty = len(data['t']) - len(ts_to_fit)
    fitted_points = np.column_stack((fit_x(ts_to_fit), fit_z(ts_to_fit)))

    distance = np.linalg.norm(fitted_points - data_points[0:len(ts_to_fit)]) 
    obj_fn_value = (distance + 0.025*interpolation_penalty) * np.exp(interpolation_penalty)
    # loss_matrix = ot.dist(data_points, fitted_points)
    # data_weights = np.ones((len(data_points),)) / len(data_points)
    # fit_weights = np.ones((len(fitted_points),)) / len(fitted_points)
    # # solves the optimal transport problem, returns minimum loss
    # distance = ot.emd2(data_weights, fit_weights, loss_matrix) * np.exp(interpolation_penalty)

    # regularise by allowing the parameter to take values close to 0, 1 or -1
    ps = np.array(parameters)
    parameter_distances = np.abs(np.column_stack((ps, ps-1, ps+1)))
    parameter_distance = np.min(parameter_distances, axis=1)
    regularisation = np.linalg.norm(parameter_distance, ord=1)
    if detailed:
        return obj_fn_value, regularisation
    return obj_fn_value + beta*(regularisation)


def data_plot(data):
    """Data visualisation function"""
    x = data['t']
    ys = [(data['x'][i], data['z'][i]) for i in range(len(x))]

    return {'x': data['x'], 'y': data['z']}
