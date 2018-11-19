class FitterReturnObject:
    """A class that defines the object that is returned from fitting"""
    def __init__(self):
        self.error_string = ""
        self.value_string = ""
        self.parameter_string = ""
        self.success_flag = False

    def is_success(self):
        return self.success_flag

    def get_errors(self):
        return self.error_string

    def get_value(self):
        return self.value_string

    def get_parameters(self):
        return self.parameter_string

    def push_error(self, error):
        self.success_flag = False
        self.error_string += error
        self.error_string += "\n"

    def push_result(self, value, parameters):
        self.value_string = str(value)
        self.parameter_string = ""
        for parameter in parameters:
            pstring = "".join((parameter['name'], ": ", str(parameter['value']), "\n"))
            self.parameter_string += pstring

    def push_failure(self, error, value, parameters):
        self.push_error(error)
        self.push_result(value, parameters)

    def push_success(self, value, parameters):
        self.success_flag = True
        self.push_result(value, parameters)


def fitter():
    return_obj = FitterReturnObject()
    try:
        print("debug")
        return_obj.push_success(0, [])
    except Exception as exception:
        return_obj.push_failure(exception, 0, [])

    return return_obj
