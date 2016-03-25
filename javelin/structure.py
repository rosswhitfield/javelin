"""This module define the Structure object"""


class Structure(object):
    def __init__(self):
        self.__positions = None
        self.__Z = []

    def __len__(self):
        return len(self.Z_array)

    def get_positions(self):
        return self.__positions

    def set_positions(self, values):
        self.__positions = values

    positions = property(get_positions, set_positions)

    def getZs(self):
        return self.__Z

    def setZs(self):
        return self.__Z

    Z = property(getZs, setZs)
