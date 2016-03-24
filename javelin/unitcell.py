import numpy as np


class UnitCell():
    def __init__(self, *args):
        self.a = 1
        self.b = 1
        self.c = 1
        self.alpha = 90
        self.beta = 90
        self.gamma = 90
        if args is not None:
            self.set_cell(args)

    def set_cell(self, *args):
        args = np.asarray(args).flatten()
        if args.size == 1:  # cubic
            self.a = self.b = self.c = np.float(args)
            self.alpha = self.beta = self.gamma = 90
        elif args.size == 3:  # orthorhombic
            self.a = np.float(args[0])
            self.b = np.float(args[1])
            self.c = np.float(args[2])
            self.alpha = self.beta = self.gamma = 90
        elif args.size == 6:
            self.a = np.float(args[0])
            self.b = np.float(args[1])
            self.c = np.float(args[2])
            self.alpha = np.float(args[3])
            self.beta = np.float(args[4])
            self.gamma = np.float(args[5])
        elif args.size == 9:  # unit cell vectors
            print(args)
        else:
            print("Invalid number of variables, unit cell unchanged")

    def get_cell(self):
        return (self.a, self.b, self.c,
                self.alpha, self.beta, self.gamma)

    cell = property(get_cell, set_cell)


def unit_cell_to_vectors(a, b, c, alpha, beta, gamma, degrees=True):
    if alpha == 90 and beta == 90 and gamma == 90:  # orthorhombic
        cell = np.diag([a, b, c])
    else:
        if degrees:
            alpha = np.radians(alpha)
            beta = np.radians(beta)
            gamma = np.radians(gamma)
        a_vec = [a, 0, 0]
        b_vec = [b * np.cos(gamma), b * np.sin(gamma), 0]
        cj_scale = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        vol = np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
                      2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        c_vec = [c * np.cos(beta),
                 c * cj_scale,
                 c * vol / np.sin(alpha)]
        cell = np.round([a_vec, b_vec, c_vec], 14)
    return cell


def unit_vectors_to_cell(cell, degrees=True):
    a = np.sqrt(np.sum(np.square(cell[0])))
    b = np.sqrt(np.sum(np.square(cell[1])))
    c = np.sqrt(np.sum(np.square(cell[2])))
    alpha = np.arccos(np.dot(cell[1], cell[2])/(b*c))
    beta = np.arccos(np.dot(cell[0], cell[2])/(a*c))
    gamma = np.arccos(np.dot(cell[0], cell[1])/(a*b))
    if degrees:
        alpha = np.degrees(alpha)
        beta = np.degrees(beta)
        gamma = np.degrees(gamma)
    return a, b, c, alpha, beta, gamma
