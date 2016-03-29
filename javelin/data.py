"""This module define the Data object"""
from javelin.unitcell import UnitCell


class Data(object):
    def __init__(self):
        self.dim = 1
        self.array = None
        self.axis_name = []
        self.axis_unit = []
        self.axis = []
        self.unit_cell = UnitCell()

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
        self.unit_cell = UnitCell(a, b, c, alpha, beta, gamma)

    def get_unit_cell(self):
        return self.unit_cell.cell
