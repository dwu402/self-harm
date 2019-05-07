from matplotlib import pyplot as plt

class View():
    def __init__(self, context):
        self.figure = None
        self.canvas = None
        self.view_function = context.visualisation_function
        self.data = context.datasets

    def __refresh__(self):
        self.figure = plt.figure()
        self.canvas = self.figure.add_subplot(111)

    def __call__(self):
        self.__refresh__()
        for data in self.data:
            self.view_function(self.canvas, data)
        self.figure.show()
