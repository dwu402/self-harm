import matplotlib.pyplot as plt


def plot_trajectory(results):
    plt.plot(results['t'], results['y'])
    plt.show()

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

def show_data(context):
    context['data_visualisation'](context['data'])
