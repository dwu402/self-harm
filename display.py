"""Module with display responsiblities

Includes plotting, writing to console and files"""
import matplotlib.pyplot as plt
import numpy as np


def new_canvas():
    """Creates a new figure and axes to plot on"""
    fig, ax = plt.subplots()
    return {'fig': fig, 'ax': ax}


def show_canvas(canvas):
    """Display the figure to screen"""
    canvas['fig'].show()
    plt.show()


def generic_plot(x, y, canvas=None, show=True, style='-'):
    """Generic plotting tool that plots onto a canvas

    Will create a new canvas if one is not provided"""
    if not canvas:
        canvas = new_canvas()
    canvas['ax'].plot(x, y, style)
    if show:
        show_canvas(canvas)

def plot_trajectory(results, canvas=None, show=True):
    """Plotter for modelled trajectories (integrates based on parameters)"""
    # test only
    x = [r[0] for r in results['y']]
    y = [r[1] for r in results['y']]
    z = [r[2] for r in results['y']]
    # test end

    generic_plot(x, z, canvas, show, style='-')


def display_parameters(parameter_obj):
    """Writes text information about fitting results to output stream (e.g. console)"""
    if parameter_obj.is_success():
        print("Fitting is successful")
    else:
        print("Fitting unsuccessful")
        print("Errors:")
        print(parameter_obj.get_errors())
    if parameter_obj.is_individual():
        print("Minimal Function Value:")
        print(parameter_obj.get_value_string())
        print("Parameter Values:")
        print(parameter_obj.get_parameter_string())
    else:
        print("Parameter Statistics")
        print(parameter_obj.get_statistics())


def show_data(context, canvas=None, show=True):
    """Plots the data that was used in fitting"""
    data_to_visualise = context['data_visualisation'](context['data'])
    generic_plot(data_to_visualise['x'], data_to_visualise['y'], canvas, show, style='-o')

def write_results(parameter_obj, filename):
    """Writes the parameter values from fitting into a file"""
    with open(filename, 'w') as fh:
        parameter_sets = parameter_obj.get_parameters()
        for parameter_set in parameter_sets:
            fh.write('---\n')
            for parameter_value in parameter_set:
                fh.write('f ' + str(parameter_value) + '\n')

def write_details(details, filename):
    """Writes details for L-Curve analysis"""
    with open(filename, 'a') as fh:
        fh.write('Beta Value: ' + str(details[0]) + '\n')
        fh.write('Residual: ' + str(details[1]) + '\n')
        fh.write('Regularisation: ' + str(details[2]) + '\n')
        fh.write('---\n')
