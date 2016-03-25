"""This module define the Data object"""


class Data(object):
    def __init__(self):
        self.dim = 1
        self.array = None
        self.axis_name = []
        self.axis_unit = []
        self.axis = []
        self.unit_cell_a = 1
        self.unit_cell_alpha = 90
        self.unit_cell_b = 1
        self.unit_cell_beta = 90
        self.unit_cell_c = 1
        self.unit_cell_gamma = 90

    def add_axis(self, name=None, units=None, array=None):
        if array is None:
            print("No axis array provided\n Not adding axis")
        else:
            self.axis.append(array)
        if name is not None:
            self.axis_name.append(name)
        if units is not None:
            self.axis_unit.append(units)

    def set_unit_cell(self, a, b, c, alpha, beta, gamma):
        self.unit_cell_a = a
        self.unit_cell_alpha = alpha
        self.unit_cell_b = b
        self.unit_cell_beta = beta
        self.unit_cell_c = c
        self.unit_cell_gamma = gamma

    def get_unit_cell(self):
        return (self.unit_cell_a,
                self.unit_cell_b,
                self.unit_cell_c,
                self.unit_cell_alpha,
                self.unit_cell_beta,
                self.unit_cell_gamma)
