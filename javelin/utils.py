"""Utility functions"""

import numpy as np


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
        cy_scale = np.cos(alpha) * (1 - np.cos(beta)) / np.sin(gamma)
        c_vec = [c * np.cos(beta),
                 c * cy_scale,
                 c * np.sqrt(np.sin(beta)**2 - cy_scale**2)]
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
