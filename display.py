import matplotlib.pyplot as plt


def new_canvas():
    fig, ax = plt.subplots()
    return {'fig': fig, 'ax': ax}


def show_canvas(canvas):
    canvas['fig'].show()
    plt.show()


def generic_plot(x, y, canvas=None, show=True, style='-'):
    if not canvas:
        canvas = new_canvas()
    canvas['ax'].plot(x, y, style)
    if show:
        show_canvas(canvas)

def plot_trajectory(results, canvas=None, show=True):
    generic_plot(results['t'], results['y'], canvas, show, style='-')


def display_parameters(parameter_obj):
    if parameter_obj.is_success():
        print("Fitting is successful")
    else:
        print("Fitting unsuccessful")
        print("Errors:")
        print(parameter_obj.get_errors())
    print("Minimal Function Value:")
    print(parameter_obj.get_value())
    print("Parameter Values:")
    print(parameter_obj.get_parameters())

def show_data(context, canvas=None, show=True):
    data_to_visualise = context['data_visualisation'](context['data'])
    generic_plot(data_to_visualise['x'], data_to_visualise['y'], canvas, show, style='-o')

def write_results(parameter_obj, filename):
    with open(filename, 'w') as fh:
        fh.write(parameter_obj.get_parameters())
