"""This module define the Structure object"""


class Fourier(object):
    __radiation_type = ["neutrons", "xray", "electrons"]

    def __init__(self):
        self.structure = None
        self.radiation = 'neutrons'
        self.wavelenght = 1.54
        self.lots = None
        self.average = 0.0
        self.na = 101
        self.no = 101
        self.ll = [0.0, 0.0, 0.0]
        self.lr = [2.0, 0.0, 0.0]
        self.ul = [0.0, 2.0, 0.0]

    def calculate(self):
        # returns Data object
        pass
