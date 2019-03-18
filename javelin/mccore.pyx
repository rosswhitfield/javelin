"""
======
mccore
======
"""

from libc.math cimport exp
from .energies cimport Energy
from .modifier cimport BaseModifier
from .random cimport random
import numpy as np
cimport cython
cimport numpy as cnp

cdef class Target:
    """Class to hold an Energy object with it associated neighbors"""
    cdef readonly int number_of_neighbors
    cdef readonly Py_ssize_t[:,:] neighbors
    cdef public Energy energy
    def __init__(self, Py_ssize_t[:,:] neighbors, Energy energy):
        assert neighbors.shape[1] == 5
        self.energy = energy
        self.neighbors = neighbors
        self.number_of_neighbors = len(self.neighbors)
    def __str__(self):
        return "{}(Energy={}\nNeighbors={})".format(self.__class__.__name__,self.energy,np.array(self.neighbors))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef (int, int, int) mcrun(BaseModifier[:] modifiers, Target[:] targets,
                    int iterations, double temperature,
                    cnp.int64_t[:,:,:,::1] a, double[:,:,:,::1] x, double[:,:,:,::1] y, double[:,:,:,::1] z):
    """This function is not meant to be used directly. It is used by
    :obj:`javelin.mc.MC`. The function does very little validation
    of the input values, it you don't provide exactly what is expected
    then segmentation fault is likely."""
    assert tuple(a.shape) == tuple(x.shape)
    assert tuple(a.shape) == tuple(y.shape)
    assert tuple(a.shape) == tuple(z.shape)
    cdef Py_ssize_t mod_x, mod_y, mod_z
    cdef int number_of_modifiers, number_of_targets, number_of_cells
    cdef int accepted_good = 0
    cdef int accepted_neutral = 0
    cdef int accepted_bad = 0
    cdef Py_ssize_t ncell, mod
    cdef Py_ssize_t[:, :] cells
    cdef Py_ssize_t target_number
    cdef double e0, e1, de
    cdef Energy energy
    cdef Target target
    cdef BaseModifier modifier
    number_of_modifiers = modifiers.shape[0]
    number_of_targets = targets.shape[0]
    mod_x = a.shape[0]
    mod_y = a.shape[1]
    mod_z = a.shape[2]
    for _ in range(iterations):
        for mod in range(number_of_modifiers):
            modifier = modifiers[mod]
            number_of_cells = modifier.number_of_cells
            cells = modifier.get_random_cells(a.shape[0], a.shape[1], a.shape[2])
            e0 = 0
            for target_number in range(number_of_targets):
                target = targets[target_number]
                energy = target.energy
                for ncell in range(number_of_cells):
                    e0 += energy.run(a, x, y, z,
                                     cells[ncell, :],
                                     target.neighbors, target.number_of_neighbors,
                                     mod_x, mod_y, mod_z)
            modifier.run(a, x, y, z)
            e1 = 0
            for target_number in range(number_of_targets):
                target = targets[target_number]
                energy = target.energy
                for ncell in range(number_of_cells):
                    e1 += energy.run(a, x, y, z,
                                     cells[ncell, :],
                                     target.neighbors, target.number_of_neighbors,
                                     mod_x, mod_y, mod_z)

            de = e1-e0
            if accept(de, temperature):
                if de < 0:
                    accepted_good += 1
                elif de == 0:
                    accepted_neutral += 1
                else:
                    accepted_bad += 1
            else:
                modifier.undo_last_run(a, x, y, z)

    return accepted_good, accepted_neutral, accepted_bad

@cython.cdivision(True)
cdef unsigned char accept(double dE, double kT):
    if dE < 0:
        return True
    elif kT <= 0:
        return False
    else:
        return random() < exp(-dE/kT)
