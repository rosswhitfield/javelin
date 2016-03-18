"""This module define the Data object"""


class Data():
    __data_types = ['Powder', "PDF", "SCDS"]
    __unit_types = ['Q', "A", "2theta", "rlu", 'd']

    def __init__(self):
        self.data_type = "Powder"
        self.dim = 1
        self.array = None
        self.axis = None
        self.units = "Q"
